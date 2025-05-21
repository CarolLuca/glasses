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
    for _ in range(num_images):
        try:
            response = requests.get(f"http://{ip_address}/capture", timeout=5)
            if response.status_code == 200:
                # Convert the image data to a numpy array
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                images.append(img)
            else:
                print(f"Failed to capture image: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"Error capturing image: {e}")
    
    return images

def interpolate_images(images):
    """Interpolate multiple images to create a clearer result."""
    if not images:
        print("No images to interpolate")
        return None
    
    if len(images) == 1:
        return images[0]
    
    # Simple averaging for interpolation
    result = np.zeros_like(images[0], dtype=np.float32)
    
    for img in images:
        result += img.astype(np.float32)
    
    result /= len(images)
    return result.astype(np.uint8)

def analyze_with_gemini(model, image):
    """Send the image to Gemini API and get analysis."""
    # Convert OpenCV BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Create prompt with instructions
    prompt = """
    Analyze the attached image and give me a response limited to maximum 75 characters. 
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

def send_analysis_to_esp32(ip_address, analysis):
    """Send the analysis result back to the ESP32."""
    try:
        url = f"http://{ip_address}/analysis"
        response = requests.post(url, data=analysis, timeout=5)
        if response.status_code == 200:
            print(f"Analysis sent to ESP32: {analysis}")
            return True
        else:
            print(f"Failed to send analysis to ESP32: HTTP {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error sending analysis to ESP32: {e}")
        return False

def main(ip_address):
    """Main function to run the continuous loop."""
    print(f"Starting camera analysis from {ip_address}")
    
    # Setup the Gemini API
    model = setup_gemini_api()
    
    try:
        while True:
            print("\nCapturing and analyzing new set of images...")
            
            # Capture multiple images
            images = capture_images(ip_address)
            
            if images:
                # Interpolate the images
                interpolated_img = interpolate_images(images)
                
                if interpolated_img is not None:
                    # Optional: Save the interpolated image for debugging
                    cv2.imwrite("latest_interpolated.jpg", interpolated_img)
                    
                    # Send to Gemini for analysis
                    analysis = analyze_with_gemini(model, interpolated_img)
                    print(f"Gemini Analysis: {analysis}")
                    
                    # Send analysis result to ESP32
                    send_analysis_to_esp32(ip_address, analysis)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze IP camera feed with Gemini AI')
    parser.add_argument('--ip', default='192.168.1.10', help='IP address of the camera (default: 192.168.1.10)')
    args = parser.parse_args()
    
    main(args.ip)