#include "esp_camera.h"
#include <WiFi.h>
#include "DFRobot_AXP313A.h"
#include <Arduino.h>
#include <U8g2lib.h>
#include <SPI.h>
#include "esp_http_server.h"


// OLED Display Configuration
#define OLED_DC  D2
#define OLED_CS  D6
#define OLED_RST D3
U8G2_SSD1309_128X64_NONAME2_1_4W_HW_SPI u8g2(/* rotation=*/U8G2_R0, /* cs=*/ OLED_CS, /* dc=*/ OLED_DC,/* reset=*/OLED_RST);

// Camera Configuration
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     45
#define SIOD_GPIO_NUM     1
#define SIOC_GPIO_NUM     2

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       46
#define Y7_GPIO_NUM       8
#define Y6_GPIO_NUM       7
#define Y5_GPIO_NUM       4
#define Y4_GPIO_NUM       41
#define Y3_GPIO_NUM       40
#define Y2_GPIO_NUM       39
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     42
#define PCLK_GPIO_NUM     5

DFRobot_AXP313A axp;

// WiFi Credentials

const char* ssid = "UPB-Guest";
const char* password = "";


// Global variables for display message
// char displayMessage[128] = "Starting...";
// bool newMessageAvailable = false;
char displayMessage[128] = "Starting...";
bool newMessageAvailable = false;

extern esp_err_t receive_analysis_handler(httpd_req_t *req);

void startCameraServer();

void setup() {
  // Initialize OLED display
  u8g2.begin();
  u8g2.setFontPosTop();

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
  
  // Initialize power management
  while(axp.begin() != 0){
    Serial.println("init error");
    delay(1000);
  }
  axp.enableCameraPower(axp.eOV2640); // Enable camera power
  
  // Configure camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // PSRAM Configuration
  if(config.pixel_format == PIXFORMAT_JPEG){
    if(psramFound()){
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
  #if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
  #endif
  }

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    strcpy(displayMessage, "Camera init failed!");
    return;
  }

  // Configure camera sensor
  sensor_t * s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  
  // Set initial frame size
  if(config.pixel_format == PIXFORMAT_JPEG){
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

  // Connect to WiFi
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  // Display connecting message
  strcpy(displayMessage, "Connecting to WiFi...");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  
  // Start camera server
  startCameraServer();
  
  // Update display with IP address
  sprintf(displayMessage, "IP: %s", WiFi.localIP().toString().c_str());
  
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

// Variables for scrolling
int scrollIndex = 0;
unsigned long lastScrollTime = 0;
void loop() {
  // Constants for timing
  const unsigned long TOTAL_DISPLAY_TIME_MS = 2000;  // Total display time (2 seconds)
  
  // Static variables to track state
  static bool messageDisplayed = false;
  static unsigned long startDisplayTime = 0;
  static int currentCharIndex = 0;
  static char currentDisplay[128] = "";
  static char previousMessage[128] = "";
  
  // Check if there's a new message to display
  if (strcmp(displayMessage, previousMessage) != 0) {
    // Reset everything when message changes
    strcpy(previousMessage, displayMessage);
    messageDisplayed = false;
    currentCharIndex = 0;
    currentDisplay[0] = '\0';
    startDisplayTime = millis();
  }
  
  // Only process if message hasn't been fully displayed
  if (!messageDisplayed) {
    unsigned long currentTime = millis();
    unsigned long elapsedTime = currentTime - startDisplayTime;
    
    // Calculate how many characters should be displayed by now
    int messageLength = strlen(displayMessage);
    if (messageLength > 0) {
      // Calculate how many characters should be shown based on elapsed time
      int charsToShow = (elapsedTime * messageLength) / TOTAL_DISPLAY_TIME_MS;
      
      // Ensure we don't exceed the message length
      charsToShow = min(charsToShow, messageLength);
      
      // If we're showing a new number of characters, update the display
      if (charsToShow > currentCharIndex) {
        // Update the current display with more characters
        strncpy(currentDisplay, displayMessage, charsToShow);
        currentDisplay[charsToShow] = '\0';
        currentCharIndex = charsToShow;
      }
      
      // Check if we've shown all characters
      if (currentCharIndex >= messageLength) {
        messageDisplayed = true;
      }
    }
    
    // Draw the current state of the message
    u8g2.firstPage();
    do {
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_profont10_mr);
      
      // Handle wrapping for display
      int messageLength = strlen(currentDisplay);
      int lines = (messageLength + 20) / 21; // Calculate lines needed
      for (int i = 0; i < min(4, lines); i++) { // Show up to 4 lines
        char lineText[22];
        int lineStart = i * 21;
        if (lineStart < messageLength) {
          int chars = min(21, messageLength - lineStart);
          strncpy(lineText, &currentDisplay[lineStart], chars);
          lineText[chars] = '\0';
          u8g2.drawStr(0, 10 + (i * 10), lineText);
        }
      }
      
      u8g2.sendBuffer();
    } while (u8g2.nextPage());
  }
  
  delay(10); // Small delay for stability
}
