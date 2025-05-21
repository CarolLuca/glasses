import os
import time
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import argparse

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def setup_gemini_api():
    """Configure the Gemini API with the API key."""
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel('gemini-2.0-flash')

def capture_images(ip_address, num_images=5):
    """Capture multiple images from the IP camera."""
    images = []
    for i in range(num_images):
        try:
            response = requests.get(f"http://{ip_address}/capture", timeout=5)
            if response.status_code == 200:
                # Convert the image data to a numpy array
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"Failed to decode image {i+1}")
                    continue
                    
                # Add to our collection
                images.append(img)
                print(f"Captured image {i+1}/{num_images}")
                
                # Brief pause between captures to allow camera to stabilize
                if i < num_images - 1:
                    time.sleep(0.2)
            else:
                print(f"Failed to capture image {i+1}: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"Error capturing image {i+1}: {e}")
    
    print(f"Successfully captured {len(images)} images")
    return images

def interpolate_images(images):
    """Align and interpolate multiple images to create a clearer, stabilized result."""
    if not images:
        print("No images to interpolate")
        return None
    
    if len(images) == 1:
        return images[0]
    
    # Convert images to grayscale for feature detection and alignment
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    # Use the first image as the reference
    reference = gray_images[0]
    aligned_images = [images[0]]
    
    # Define the feature detector
    orb = cv2.ORB_create(nfeatures=1500)
    
    # For each image after the first one
    for i in range(1, len(images)):
        # Find keypoints and descriptors for the reference and current image
        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(gray_images[i], None)
        
        # If no features found, skip alignment for this image
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Not enough features in image {i}, using without alignment")
            aligned_images.append(images[i])
            continue
            
        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use only good matches
        good_matches = matches[:min(40, len(matches))]
        
        if len(good_matches) < 4:
            print(f"Not enough good matches in image {i}, using without alignment")
            aligned_images.append(images[i])
            continue
        
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print(f"Could not find homography for image {i}, using without alignment")
            aligned_images.append(images[i])
            continue
        
        # Apply homography to align the image
        h, w = reference.shape
        aligned = cv2.warpPerspective(images[i], M, (w, h))
        aligned_images.append(aligned)
    
    # Convert images to float32 for processing
    aligned_np = [img.astype(np.float32) for img in aligned_images]
    
    # Calculate median of aligned images (more robust than mean)
    stacked = np.stack(aligned_np, axis=0)
    result_median = np.median(stacked, axis=0)
    
    # Calculate mean of aligned images
    result_mean = np.mean(stacked, axis=0)
    
    # Mix median and mean for better results (median reduces noise, mean preserves details)
    result = cv2.addWeighted(result_median, 0.6, result_mean, 0.4, 0)
    
    return result.astype(np.uint8)

def analyze_with_gemini(model, image):
    """Send the image to Gemini API and get analysis."""
    # Convert OpenCV BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Create prompt with instructions
    prompt = """
    Analyze the attached image and give me a response limited to maximum 100 characters. 
    If the sent image has a lot of text and it seems to be containing a question, provide an answer to the respective question. 
    If not, then there are two paths. If what you see seems to be something branded, try to mention the brand as well. 
    Otherwise just describe what you see keeping in mind the character limit previously imposed.
    """
    
    try:
        # Pass the PIL image directly to the API
        response = model.generate_content([prompt, pil_image])
        return response.text
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return f"Error: {str(e)}"

def main(ip_address):
    """Main function to run the continuous loop."""
    print(f"Starting camera analysis from {ip_address}")
    
    # Setup the Gemini API
    model = setup_gemini_api()
    
    # Create output directory for debug images if it doesn't exist
    os.makedirs("debug_images", exist_ok=True)
    
    # Create an empty array to store the history of processed images
    image_history = []
    history_max_size = 3  # Keep track of the last 3 good images
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            print(f"Capturing and analyzing new set of images from {ip_address}...")
            
            # Capture multiple images
            images = capture_images(ip_address)
            
            if images:
                # Save the first raw image for debugging
                cv2.imwrite(f"debug_images/raw_{iteration}.jpg", images[0])
                
                # Interpolate the images
                interpolated_img = interpolate_images(images)
                
                if interpolated_img is not None:
                    # Save the interpolated image for debugging
                    cv2.imwrite(f"debug_images/stabilized_{iteration}.jpg", interpolated_img)
                    
                    # Apply image enhancement
                    enhanced = enhance_image(interpolated_img)
                    cv2.imwrite(f"debug_images/enhanced_{iteration}.jpg", enhanced)
                    
                    # Add to history
                    image_history.append(enhanced)
                    if len(image_history) > history_max_size:
                        image_history.pop(0)  # Remove oldest image
                    
                    # If we have enough history, perform temporal averaging
                    if len(image_history) >= 2:
                        # Stack images for temporal processing
                        temporal_stack = np.stack(image_history, axis=0)
                        
                        # Apply weighted temporal averaging (more weight to recent frames)
                        weights = np.linspace(0.5, 1.0, len(image_history))
                        weights = weights / np.sum(weights)  # Normalize weights
                        
                        # Apply weighted sum across temporal dimension
                        temporal_result = np.zeros_like(image_history[0], dtype=np.float32)
                        for i, img in enumerate(image_history):
                            temporal_result += img.astype(np.float32) * weights[i]
                        
                        final_image = temporal_result.astype(np.uint8)
                        cv2.imwrite(f"debug_images/temporal_{iteration}.jpg", final_image)
                    else:
                        final_image = enhanced
                    
                    # Send to Gemini for analysis
                    print("Sending to Gemini API for analysis...")
                    analysis = analyze_with_gemini(model, final_image)
                    print(f"Gemini Analysis: {analysis}")
                else:
                    print("Failed to create interpolated image")
            else:
                print("No images were captured")
            
            # Wait before next iteration
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def enhance_image(image):
    """Apply advanced enhancements to improve image quality using NumPy operations."""
    # Convert to float32 for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Split into channels for separate processing
    b, g, r = cv2.split(img_float)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Convert back to 8-bit unsigned integers for CLAHE
    b_uint8 = (b * 255).astype(np.uint8)
    g_uint8 = (g * 255).astype(np.uint8)
    r_uint8 = (r * 255).astype(np.uint8)
    
    # Apply CLAHE to each channel
    b_eq = clahe.apply(b_uint8)
    g_eq = clahe.apply(g_uint8)
    r_eq = clahe.apply(r_uint8)
    
    # Convert back to float32 [0-1]
    b_eq_float = b_eq.astype(np.float32) / 255.0
    g_eq_float = g_eq.astype(np.float32) / 255.0
    r_eq_float = r_eq.astype(np.float32) / 255.0
    
    # Merge the enhanced channels
    enhanced_float = cv2.merge([b_eq_float, g_eq_float, r_eq_float])
    
    # Apply bilateral filter for edge-preserving smoothing (using float [0-1])
    bilateral = cv2.bilateralFilter(enhanced_float, 9, 75/255, 75/255)
    
    # Apply unsharp mask for sharpening
    # Create a gaussian blur version
    gaussian = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
    # Subtract blurred from original and add back to original
    unsharp_mask = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)
    
    # Clip values to [0, 1]
    clipped = np.clip(unsharp_mask, 0, 1)
    
    # Convert back to uint8 for further processing
    enhanced_uint8 = (clipped * 255).astype(np.uint8)
    
    # Apply a small amount of denoise as a final step
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_uint8, None, 5, 5, 7, 21)
    
    # Apply Gamma correction to enhance mid-tones
    gamma = 1.1  # Slight boost
    gamma_corrected = np.power(denoised / 255.0, 1.0/gamma) * 255.0
    
    return gamma_corrected.astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze IP camera feed with Gemini AI')
    parser.add_argument('--ip', default='192.168.1.10', help='IP address of the camera (default: 192.168.1.10)')
    args = parser.parse_args()
    
    main(args.ip)