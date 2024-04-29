#include <Arduino_FreeRTOS.h>
#include <semphr.h>
#include <task.h>
#include <event_groups.h>
#include <DetectionsBuffer.h>
#include <Dictionary.h>
#include <SoftwareSerial.h>
#include <MotorDriver.h>
#include "TimeManagement.h"
#include "robot_control.h"

/**
 IMPORTANT NOTICE! (Yes, this is actually important)
 Seriously read this. If you don't read and follow this step your code will not compile.
 If you ask me about it and I find you did not read this I will laugh at you.

 For this code to run you need to go to FreeRTOSConfig.h in the FreeRTOS directory in your
 Arduino Libraries Directory, normally located in the Documents folder of Windows Machines,
 Documents/Arduino/libraries/FreeRTOS/FreeRTOSConfig.h and add this line to the 
 #define INCLUDE Statements:

 #define INCLUDE_eTaskGetState               	1

 I usually put it under the group beneath this comment:
 Set the following definitions to 1 to include the API function, or zero
 to exclude the API function.

 Now your code will compile. Good Job.

 --Jordan

**/


/**
 Note to everyone, this is the syntax for creating tasks

 BaseType_t xTaskCreate(TaskFunction_t pvTaskCode,
                       const char * const pcName,
                       unsigned short usStackDepth,
                       void *pvParameters,
                       UBaseType_t uxPriority,
                       TaskHandle_t *pvCreatedTask);

**/


TimeManagement timeManager;


RobotControl rc;

// Enum for different robot states
enum RobotState {
  WAIT_FOR_START,
  GET_BIG_BOXES,
  GET_SMALL_BOXES,
  DEPOSIT_BIG_BOXES,
  DEPOSIT_SMALL_BOXES,
  FOLLOW_LINE,  // Has Counter
  GO_TO_RED_ZONE,
  GO_TO_BLUE_ZONE,
  GO_TO_GREEN_ZONE,
  GET_ROCKETS,
  CROSS_GAP,
  DEPOSIT_ROCKETS,
  PUSH_BUTTON,
  DISPLAY_LOGO,
  DONE,
  EMERGENCY_STOP
};

// RobotState Vals
RobotState prevState = WAIT_FOR_START;
RobotState currentState = WAIT_FOR_START;
int Follow_Line_Counter = 0;

const int lightSensorPin = A0;


// Obj Detection Variables
#define BUFFER_SIZE 512          // Maximum size of String that can be passed from Jetson. 
#define MAX_CLASSNAME_LENGTH 15  // Maximum size of the class name char array
#define MAX_DIRECTION_LENGTH 6   // Maximum size of the direction char array. 

char dataBuffer[BUFFER_SIZE];
unsigned long previousMillis = 0;  // Stores the last time a request was made
const long interval = 10000;

Dictionary &class_names_dict = *(new Dictionary(11));
Dictionary &dir_names_dict = *(new Dictionary(2));
Dictionary &class_names_rev = *(new Dictionary(11));


// Debug mode 
const int debugPin = 10;  // The "Oh no, what did I break now?" pin
bool debugMode = false;   
volatile bool printDebugFlag = false;


// RTOS Vals
// Emergency Stop Pin - The "Oh no, everything's on fire" button
const int emergencyStopPin = 51;


// Task Handlers
TaskHandle_t readDetTaskHandle;
TaskHandle_t processDetTaskHandle;
TaskHandle_t SensorBoxTaskHandle;
TaskHandle_t debugTaskHandle;



// Mutex for RobotState
SemaphoreHandle_t stateMutex;
// Buffer Mutex
SemaphoreHandle_t bufferMutex;
// Serial Mutex
SemaphoreHandle_t serialMutex;

// Event Group for Detections
#define BIT_NEW_DATA_AVAILABLE (1 << 0)
#define BIT_READ_DETECTIONS (1 << 1)
EventGroupHandle_t xDetectionsEventGroup;
// Tasks
void MotorBoxStateManagement(void *pvParameters);
void SensorBox(void *pvParameters);
void readDetTask(void *pvParameters);
void processDetTask(void *pvParameters);
void DebugBox(void *pvParameters);
void StateTimeoutCallback(TimerHandle_t xTimer);


void setup() {


  Serial.begin(115200);
  Serial2.begin(115200);
  Serial.println("Setup");
  pinMode(lightSensorPin, INPUT);
  rc.init();
  clearBuffer();
  String dir_json = "{\"0\": \"left\", \"1\": \"right\"}";
  String class_json = "{\"0\": \"BigBox\", \"1\": \"BlueZone\", \"2\": \"Button\", \"3\": \"GreenZone\", \"4\": \"Nozzle\", \"5\": \"RedZone\", \"6\": \"Rocket\", \"7\": \"SmallBox\", \"8\": \"StartZone\", \"9\": \"WhiteLine\", \"10\": \"YellowLine\"}";
  String class_rev = "{\"BigBox\": \"0\", \"BlueZone\": \"1\", \"Button\": \"2\", \"GreenZone\": \"3\", \"Nozzle\": \"4\", \"RedZone\": \"5\", \"Rocket\": \"5\", \"SmallBox\": \"7\", \"StartZone\": \"8\", \"WhiteLine\": \"9\", \"YellowLine\": \"10\"}";
  dir_names_dict.jload(dir_json);
  class_names_dict.jload(class_json);
  class_names_rev.jload(class_rev);

  // Create a mutex for state variable and serial
  stateMutex = xSemaphoreCreateMutex();
  bufferMutex = xSemaphoreCreateMutex();

  pinMode(emergencyStopPin, INPUT_PULLUP);     // Set as input with pull-up
  pinMode(debugPin, INPUT_PULLUP);             // Set debug pin as input with pull-up
  debugMode = (digitalRead(debugPin) == LOW);  // Check if the pin is LOW (switch closed)
  // Serial.println(debugMode);

  attachInterrupt(digitalPinToInterrupt(emergencyStopPin), emergencyStopISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(debugPin), debugModeISR, FALLING);


  xDetectionsEventGroup = xEventGroupCreate();

  Serial.println("Setup");
  // Creating our cast of tasks - it's like a talent show, but with more crashing
  xTaskCreate(MotorBoxStateManagement, "MotorBoxStateManagement", 1000, NULL, 1, &timeManager.MotorBoxTaskHandle);
  xTaskCreate(SensorBox, "SensorBox", 128, NULL, 2, &SensorBoxTaskHandle);
  xTaskCreate(readDetTask, "readDetTask", 500, NULL, 4, &readDetTaskHandle);
  xTaskCreate(processDetTask, "processDetTask", 500, NULL, 3, &processDetTaskHandle);

  if (debugMode) {
    xTaskCreate(DebugBox, "DebugBox", 200, NULL, 5, &debugTaskHandle);
  }
  Serial.println("Setup Done");
  // Don't create both. you will break things.
  // if (debugMode) {
  //   xTaskCreate(SensorBox, "SensorBox", 1000, NULL, 5, &SensorBoxTaskHandle);
  //   xTaskCreate(readDetTask, "readDetTask", 1000, NULL, 4, &readDetTaskHandle);
  //   xTaskCreate(processDetTask, "processDetTask", 1000, NULL, 3, &processDetTaskHandle);
  //   xTaskCreate(MotorBoxStateManagement, "MotorBoxStateManagement", 128, NULL, 2, &MotorBoxTaskHandle);
  //   xTaskCreate(DebugBox, "DebugBox", 200, NULL, 1, &debugTaskHandle);
  //  } else {
  //     xTaskCreate(SensorBox, "SensorBox", 1000, NULL, 4, &SensorBoxTaskHandle);
  //     xTaskCreate(readDetTask, "readDetTask", 1000, NULL, 3, &readDetTaskHandle);
  //     xTaskCreate(processDetTask, "processDetTask", 1000, NULL, 2, &processDetTaskHandle);
  //     xTaskCreate(MotorBoxStateManagement, "MotorBoxStateManagement", 128, NULL, 1, &MotorBoxTaskHandle);
  //  }
}



// LEAVE THIS EMPTY. NO TOUCHING. AT ALL. UNDER ANY CIRCUMSTANCES. JUST DON'T DO IT.
// WITH THE WAY THIS CODE IS SET UP WE WILL NEVER REACH THIS SECTION OF THE CODE.
// -- Jordan
void loop() {
}


void MotorBoxStateManagement(void *pvParameters) {
  uint32_t notificationValue;
  const uint32_t MY_ULONG_MAX = 0xFFFFFFFF;

  for (;;) {
    Serial.println("Motorbox");
    if (xTaskNotifyWait(0x00, MY_ULONG_MAX, &notificationValue, pdMS_TO_TICKS(100)) == pdTRUE) {
      bool timeout = notificationValue != 0;
      // State completed
      if (!timeout) {
        Serial.println("State completed normally");
      } else {
        Serial.println("We Timed out.");
      }
    }
    if (currentState != prevState) {
      Serial.println("Clearing Buffer");
      xSemaphoreTake(bufferMutex, portMAX_DELAY);
      clearBuffer();
      xSemaphoreGive(bufferMutex);
      if (timeManager.timeOut) {
        timeManager.timeOut = false;
      }
    }
    switch (currentState) {
      case WAIT_FOR_START:
        // Code to handle waiting for start
        timeManager.startState(0);
        wait_for_start();
        break;
      case GET_BIG_BOXES:
        // Code for getting big blocks
        timeManager.startState(1);
        get_big_boxes();
        break;
      case GET_SMALL_BOXES:
        // Code for getting small blocks
        timeManager.startState(2);
        get_small_boxes();
        break;
      case DEPOSIT_BIG_BOXES:
        // Code for depositing big blocks
        timeManager.startState(7);
        deposit_big_boxes();
        break;
      case DEPOSIT_SMALL_BOXES:
        // Code for depositing small blocks
        timeManager.startState(5);
        deposit_small_boxes();
        break;
      case FOLLOW_LINE:
        // Code to follow the yellow line
        switch (Follow_Line_Counter) {
          case 0:
            timeManager.startState(3);
            break;
          case 1:
            timeManager.startState(8);
            break;
          case 2:
            timeManager.startState(11);
            break;
          case 3:
            timeManager.startState(13);
            break;
          default:
            break;
        }

        follow_line();
        break;

      case GO_TO_RED_ZONE:
        // Code to go to the red zone
        timeManager.startState(4);
        go_to_red_zone();
        break;
      case GO_TO_BLUE_ZONE:
        // Code to go to the blue zone
        timeManager.startState(6);
        go_to_blue_zone();
        break;
      case GO_TO_GREEN_ZONE:
        // Code to go to the green zone
        timeManager.startState(9);
        go_to_green_zone();
        break;
      case GET_ROCKETS:
        // Code for getting rockets
        timeManager.startState(10);
        get_rockets();
        break;
      case DEPOSIT_ROCKETS:
        // Code for depositing rockets
        timeManager.startState(14);
        deposit_rockets();
        break;
      case CROSS_GAP:
        // Code to cross the gap
        timeManager.startState(12);
        cross_gap();
        break;
      case PUSH_BUTTON:
        // Code to push stop timer button
        timeManager.startState(16);
        push_button();
        break;
      case DISPLAY_LOGO:
        // Code to display the logo
        timeManager.startState(15);
        display_logo();
        break;
      case DONE:
        // Code for stopping when all tasks completed
        timeManager.startState(17);
        done();
        break;
      case EMERGENCY_STOP:
        // Code for emergency stop
        timeManager.startState(18);
        emergency_stop();
        break;
      default:
        break;
    }
    prevState = currentState;
  }
}

void SensorBox(void *pvParameters) {
  for (;;) {
    if (currentState != DONE || currentState != EMERGENCY_STOP) {
      // Serial.println("Sensor Task");
      vTaskDelay(300 / portTICK_PERIOD_MS);  // On for 3 seconds
    }
  }
}



void readDetTask(void *pvParameters) {
  for (;;) {  // Infinite loop for the task
    EventBits_t uxBits = xEventGroupWaitBits(
      xDetectionsEventGroup,
      BIT_READ_DETECTIONS,
      pdTRUE,          // Clear BIT_READ_DETECTIONS on exit.
      pdFALSE,         // Wait for just BIT_READ_DETECTIONS.
      portMAX_DELAY);  // Wait indefinitely.
    if (currentState != DONE || currentState != EMERGENCY_STOP) {
      if ((uxBits & BIT_READ_DETECTIONS) != 0) {

        vTaskDelay(pdMS_TO_TICKS(50));
        // Serial.println("RequestDetTask");
        Serial2.println("REQUEST");
        // Wait for a response with a timeout
        unsigned long startTime = millis();
        while (!Serial2.available() && millis() - startTime < 1000) {
          // Waiting for response with 5 seconds timeout
          vTaskDelay(pdMS_TO_TICKS(10));  // Small delay to prevent blocking CPU
        }

        // Read and store the response
        if (Serial2.available()) {
          String data = Serial2.readStringUntil('\n');
          data.toCharArray(dataBuffer, BUFFER_SIZE);
        }

        xEventGroupSetBits(xDetectionsEventGroup, BIT_NEW_DATA_AVAILABLE);
      }
    }
    taskYIELD();
  }
}


void processDetTask(void *pvParameters) {
  for (;;) {
    EventBits_t uxBits = xEventGroupWaitBits(
      xDetectionsEventGroup,
      BIT_NEW_DATA_AVAILABLE,
      pdTRUE,          // Clear BIT_NEW_DATA_AVAILABLE on exit.
      pdFALSE,         // Wait for just BIT_NEW_DATA_AVAILABLE.
      portMAX_DELAY);  // Wait indefinitely.
    if (currentState != DONE || currentState != EMERGENCY_STOP) {
      if ((uxBits & BIT_NEW_DATA_AVAILABLE) != 0) {
        vTaskDelay(pdMS_TO_TICKS(50));
        // Serial.println("ProcessDetTask");

        // Process the data in dataBuffer
        // Serial.println("Received Detections");
        // Serial.println(dataBuffer);
        xSemaphoreTake(bufferMutex, portMAX_DELAY);
        processDetections(dataBuffer);
        xSemaphoreGive(bufferMutex);
        xEventGroupSetBits(xDetectionsEventGroup, BIT_READ_DETECTIONS);
      }
    }
    // Yield to other tasks
    taskYIELD();
  }
}

void DebugBox(void *pvParameters) {
  for (;;) {
    // if (printDebugFlag) {

    if (currentState != DONE || currentState != EMERGENCY_STOP) {

      vTaskDelay(pdMS_TO_TICKS(50));  // FreeRTOS delay
      Serial.println("PrintDebug");

      // vTaskDelay(pdMS_TO_TICKS(2000)); // FreeRTOS delay
      eTaskState readtaskState = eTaskGetState(readDetTaskHandle);
      eTaskState processtaskState = eTaskGetState(processDetTaskHandle);
      // Serial.println(readtaskState);
      // Serial.println(processtaskState);

      if ((readtaskState <= 2) || (processtaskState <= 2)) {
        Serial.println("DETECTIONS ON");
        printDetections();
      }
      vTaskDelay(pdMS_TO_TICKS(2000));  // FreeRTOS delay
    }

    // Yield to other tasks
    taskYIELD();
  }
}






void processDetections(char data[]) {
  // Tokenize the data string into individual detections using strtok
  char *detection = strtok(data, ";");
  while (detection != NULL) {
    // Serial.println(detection);
    parseDetection(detection);
    detection = strtok(NULL, ";");  // Get next detection
  }
}


void parseDetection(char *detection) {
  char class_name[MAX_CLASSNAME_LENGTH];
  int class_key, dir_key;
  float confidence;
  float depth_mm;
  float depth_in;
  float x, y, z;
  float horizontal_angle, timestamp;
  char direction[MAX_DIRECTION_LENGTH];
  char *token;
  char *rest = detection;

  token = strtok_r(rest, ",", &rest);
  if (token != NULL) {
    class_key = atoi(token);
  }

  // Continue for the rest of the fields
  token = strtok_r(rest, ",", &rest);
  if (token != NULL) {
    timestamp = atof(token);
  }

  // Continue for the rest of the fields
  token = strtok_r(rest, ",", &rest);
  if (token != NULL) {
    depth_mm = atof(token);
  }

  token = strtok_r(rest, ",", &rest);
  if (token != NULL) {
    x = atof(token);
  }


  token = strtok_r(rest, ",", &rest);
  if (token != NULL) {
    horizontal_angle = atof(token);
  }

  token = strtok_r(rest, ",", &rest);
  if (token != NULL) {
    dir_key = atoi(token);
    ;
  }

  String ck(class_key);
  String dk(dir_key);

  String class_n = class_names_dict[ck];
  String dir_n = dir_names_dict[dk];

  class_n.toCharArray(class_name, MAX_CLASSNAME_LENGTH);
  dir_n.toCharArray(direction, MAX_DIRECTION_LENGTH);

  Detection newDetection(class_name, timestamp, depth_mm, x, horizontal_angle, direction);
  // if (!debugMode) {
  //   printDetection(newDetection);
  // }

  addDetectionToBuffer(newDetection);
}


void printDetections() {
  // Print the closest detection
  Detection closest = getClosestDetection();
  Serial.println("Closest Detection:");
  printDetection(closest);

  // // Print the latest detection
  // Detection latest = getLatestDetection();
  // Serial.println("Latest Detection:");
  // printDetection(latest);

  // // Loop through and print all detections
  // Serial.println("All Detections:");
  // for (int i = 0; i < getBufferSize(); i++) {
  //     Detection d = getDetectionFromBuffer(i);
  //     printDetection(d);
  // }
}



void printDetection(const Detection &d) {
  if (strlen(d.class_name) > 0) {  // Check if the detection is valid
    Serial.print("Class Name: ");
    Serial.println(d.class_name);
    Serial.print("Timestamp: ");
    Serial.println(d.timestamp, 2);
    Serial.print("Depth MM: ");
    Serial.println(d.depth_mm);
    Serial.print("X Component: ");
    Serial.println(d.x);
    Serial.print("Horizontal Angle: ");
    Serial.println(d.horizontal_angle);
    Serial.print("Direction: ");
    Serial.println(d.direction);
    Serial.println("-------------------");
  } else {
    Serial.println("No Detection Data");
  }
}


void emergencyStopISR() {
  currentState = EMERGENCY_STOP;
  Serial.println("Emergency Stop");
  // Disconnect Motors Here
}

void debugModeISR() {
  debugMode = true;
  Serial.println("Debug Mode On");
  // Disconnect Motors Here
}


// wait_for_start
// State Number 0
// Current Max Time 0 seconds (Doesn't have time limit)
void wait_for_start() {
  int upperThreshold = 180;
  int lowerThreshold = 100;
  xEventGroupSetBits(xDetectionsEventGroup, BIT_READ_DETECTIONS);
  vTaskResume(readDetTaskHandle);
  vTaskResume(processDetTaskHandle);
  vTaskDelay(100 / portTICK_PERIOD_MS);
  int lightSensorValue = analogRead(lightSensorPin);
  Serial.print("Light Sensor Value: ");
  Serial.println(lightSensorValue);
  Serial2.println("WAIT_FOR_START");
  Serial.println("Waiting to Start");
  lightSensorValue = analogRead(lightSensorPin);
  Serial.print("Light Sensor Value: ");
  Serial.println(lightSensorValue);
  // while (lightSensorValue > upperThreshold) {
  //   lightSensorValue = analogRead(lightSensorPin);
  //   Serial.print("Light Sensor Value: ");
  //   Serial.println(lightSensorValue);
  //   vTaskDelay(50 / portTICK_PERIOD_MS);
  // }
  currentState = GET_BIG_BOXES;
  vTaskDelay(5000 / portTICK_PERIOD_MS);
  timeManager.endState(0, false);
}

// get_big_boxes
// State Number 1
// Current Max Time 10 seconds
void get_big_boxes() {
  bool stateComplete = false;
  Detection most_recent;
  vTaskResume(readDetTaskHandle);
  vTaskResume(processDetTaskHandle);
  Serial.println(stateComplete);
  Serial2.println("GET_BIG_BOXES");
  rc.myservo.write(127);
  rc.theGobbler(1);
  rc.pickup();
  while ((timeManager.getRemainingTimeForState(1) > 100) && !stateComplete && !timeManager.timeOut) {
    // vTaskDelay(100 / portTICK_PERIOD_MS);
    Serial.println("Getting Big Boxes");
    Serial.println(timeManager.getRemainingTimeForState(1));

    for (int i = 0; i < 2; i++) {
      if (timeManager.getRemainingTimeForState(1) < 800) {
        timeManager.timeOut = true;
        break;
      }
      rc.motorDriver.setSpeed(50, 50);
      Serial.println(rc.USDistance());
      while (rc.USDistance() > 11) {
        rc.motorDriver.startMove();
        Serial.println(rc.USDistance());
        vTaskDelay(1 / portTICK_PERIOD_MS);
      }

      rc.motorDriver.setSpeed(50, 50);
      for (int i = 0; i < 500; i++) {
        rc.motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
      }

      rc.motorDriver.setSpeed(-50, -250);
      for (int i = 0; i < 600; i++) {
        rc.motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
      }

      rc.motorDriver.setSpeed(-250, -50);
      for (int i = 0; i < 600; i++) {
        rc.motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
      }

      rc.motorDriver.setSpeed(0, 0);
      for (int i = 0; i < 500; i++) {
        rc.motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
      }
    }
    if (timeManager.getRemainingTimeForState(1) < 600) {
      break;
    }
    rc.motorDriver.setSpeed(50, 50);
    //Serial.println(rc.USDistance());
    while (rc.USDistance() > 11) {
      rc.motorDriver.startMove();
      Serial.println(rc.USDistance());
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    rc.motorDriver.setSpeed(0, 0);
    for (int i = 0; i < 500; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    rc.motorDriver.setSpeed(-200, -200);
    for (int i = 0; i < 500; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    rc.myservo.write(115);
    rc.theGobbler(0);

    rc.motorDriver.setSpeed(0, 0);
    for (int i = 0; i < 500; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }
    stateComplete = true;
    break;
  }

  rc.TurnLeft90();
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(1, stateComplete);
  xSemaphoreGive(bufferMutex);
  currentState = GET_SMALL_BOXES;
}


// get_small_boxes
// State Number 2
// Current Max Time 10 seconds
void get_small_boxes() {
  bool stateComplete = false;
  vTaskDelay(250 / portTICK_PERIOD_MS);
  Serial2.println("GET_SMALL_BOXES");
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  while ((timeManager.getRemainingTimeForState(2) > 0) && !stateComplete && !timeManager.timeOut) {
    rc.myservo.write(127);
    rc.theGobbler(1);
    Serial.println(timeManager.getRemainingTimeForState(2));
    Serial.println("Getting Small Boxes");
    for (int i = 0; i < 3; i++) {
      Serial.print("Box ");
      Serial.println(i);
      if (timeManager.getRemainingTimeForState(1) < 80) {
        timeManager.timeOut = true;
        break;
      }
      Detection smallBox = getLeftmostDetection();
      bool grabbed = false;
      Serial.print("Distance ");
      Serial.println(smallBox.depth_mm);
      Serial.print("Angle ");
      Serial.println(smallBox.horizontal_angle);
      if (!grabbed) {
        if (timeManager.getRemainingTimeForState(1) < 180) {
          timeManager.timeOut = true;
          break;
        }
        smallBox = getLeftmostDetection();
        while (!rc.turnTo(smallBox)) {
          Serial.println("Turning");
          smallBox = getLeftmostDetection();
          Serial.print("Distance ");
          Serial.println(smallBox.depth_mm);
          Serial.print("Angle ");
          Serial.println(smallBox.horizontal_angle);
          vTaskDelay(10 / portTICK_PERIOD_MS);
        }
        Serial.println("Done Turning");
        while (!rc.moveTo(smallBox)) {
          Serial.println("Moving");
          smallBox = getLeftmostDetection();
          Serial.print("Distance ");
          Serial.println(smallBox.depth_mm);
          Serial.print("Angle ");
          Serial.println(smallBox.horizontal_angle);
          vTaskDelay(10 / portTICK_PERIOD_MS);
        }
        Serial.println("Done Moving");
        grabbed = true;
      }
    }
    // timeManager.endState(2);
  }
  rc.myservo.write(115);
  rc.theGobbler(0);
  rc.cruisin();
  currentState = FOLLOW_LINE;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(2, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// follow_line 
// State Numbers 3, 8, 11, 13
// Current Max Times 7, 5, 5, and 5 seconds respectively
void follow_line() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskSuspend(readDetTaskHandle);
  vTaskSuspend(processDetTaskHandle);
  Serial2.print("FOLLOW_LINE.");
  Serial2.println(Follow_Line_Counter);


  switch (Follow_Line_Counter) {
    case 0:
      line_follow(3, DEPOSIT_ROCKETS, Follow_Line_Counter, stateComplete);
      break;
    case 1:
      line_follow(8, GO_TO_GREEN_ZONE, Follow_Line_Counter, stateComplete);
      break;
    case 2:
      line_follow(11, CROSS_GAP, Follow_Line_Counter, stateComplete);
      break;
    case 3:
      line_follow(13, DEPOSIT_ROCKETS, Follow_Line_Counter, stateComplete);
      break;
    default:
      break;
  }
  Follow_Line_Counter++;
}



// deposit_big_boxes
// State Number 7
// Current Max Time 3 seconds
void deposit_big_boxes() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskSuspend(readDetTaskHandle);
  vTaskSuspend(processDetTaskHandle);
  Serial2.println("DEPOSIT_BIG_BOXES");
  while ((timeManager.getRemainingTimeForState(7) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(7));
    Serial.println("Depositing Big Boxes");
    rc.myservo.write(127);
    rc.theGobbler(-1);
    rc.pickup();
  }
  rc.myservo.write(115);
  rc.theGobbler(0);
  rc.TurnLeft45();
  currentState = FOLLOW_LINE;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(7, stateComplete);
  xSemaphoreGive(bufferMutex);
}


// deposit_small_boxes
// State Number 5
// Current Max Time 3 seconds
void deposit_small_boxes() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskSuspend(readDetTaskHandle);
  vTaskSuspend(processDetTaskHandle);
  Serial2.println("DEPOSIT_SMALL_BOXES");
  while ((timeManager.getRemainingTimeForState(5) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(5));
    Serial.println("Depositing Small Boxes");
    rc.myservo.write(127);
    rc.theGobbler(-1);
  }
  rc.myservo.write(115);
  rc.theGobbler(0);
  rc.cruisin();
  currentState = GO_TO_BLUE_ZONE;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(5, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// go_to_red_zone
// State Number 4
// Current Max Time 4 seconds
void go_to_red_zone() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskResume(readDetTaskHandle);
  vTaskResume(processDetTaskHandle);
  xEventGroupSetBits(xDetectionsEventGroup, BIT_READ_DETECTIONS);
  vTaskDelay(250 / portTICK_PERIOD_MS);
  Detection redZone = getLatestDetection();
  bool grabbed = false;
  Serial2.println("GO_TO_RED_ZONE");
  while ((timeManager.getRemainingTimeForState(4) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(4));
    Serial.println("Going to Red Zone");
    while (!grabbed) {
      if (timeManager.getRemainingTimeForState(1) < 180) {
        timeManager.timeOut = true;
        break;
      }
      redZone = getLatestDetection();
      while (!rc.turnTo(redZone)) {
        redZone = getLatestDetection();
      }
      while (!rc.moveTo(redZone)) {
        redZone = getLatestDetection();
      }
      grabbed = true;
    }
    break;
  }

  currentState = DEPOSIT_SMALL_BOXES;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(4, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// go_to_blue_zone
// State Number 6
// Current Max Time 2 seconds
void go_to_blue_zone() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskResume(readDetTaskHandle);
  vTaskResume(processDetTaskHandle);
  Serial2.println("GO_TO_BLUE_ZONE");
  rc.TurnLeft90();
  rc.moveForward();
  stateComplete = true;
  currentState = DEPOSIT_BIG_BOXES;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(6, stateComplete);
  xSemaphoreGive(bufferMutex);
}


// go_to_green_zone
// State Number 9
// Current Max Time 3 seconds
void go_to_green_zone() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskResume(readDetTaskHandle);
  vTaskResume(processDetTaskHandle);
  xEventGroupSetBits(xDetectionsEventGroup, BIT_READ_DETECTIONS);
  vTaskDelay(250 / portTICK_PERIOD_MS);
  Serial2.println("GO_TO_GREEN_ZONE");
  rc.TurnRight45();
  Detection greenZone = getLatestDetection();
  bool grabbed = false;
  while ((timeManager.getRemainingTimeForState(9) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(9));
    Serial.println("Going to Green Zone");
    while (!grabbed) {
      if (timeManager.getRemainingTimeForState(1) < 180) {
        timeManager.timeOut = true;
        break;
      }
      greenZone = getLatestDetection();
      while (!rc.turnTo(greenZone)) {
        greenZone = getLatestDetection();
      }
      while (!rc.moveTo(greenZone)) {
        greenZone = getLatestDetection();
      }
      grabbed = true;
    }
    break;
  }
  currentState = GET_ROCKETS;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(9, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// get_rockets
// State Number 10
// Current Max Time 10 seconds
void get_rockets() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  Serial2.println("GET_ROCKETS");
  while ((timeManager.getRemainingTimeForState(10) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(10));
    Serial.println("Getting Rockets");
    //rc.doTheJig();
  }
  currentState = FOLLOW_LINE;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(10, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// deposit_rockets
// State Number 14
// Current Max Time 16 seconds
void deposit_rockets() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  vTaskResume(readDetTaskHandle);
  vTaskResume(processDetTaskHandle);
  Serial2.println("DEPOSIT_ROCKETS");
  while ((timeManager.getRemainingTimeForState(14) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(14));
    Serial.println("Depositing Rockets");
    // if (rc.takeItBack()) {
    //   if (rc.startTheMotors()) {
    //     if (timeManager.getRemainingTimeForState(1) < 1500) {
    //       if (rc.stopTheMotors()) {
    //         stateComplete = true;
    //         break;
    //       }
    //     }
    //   }
    // }
  }
  currentState = DISPLAY_LOGO;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(14, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// cross_gap
// State Number 12
// Current Max Time 16 seconds
void cross_gap() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  Serial2.println("CROSS_GAP");
  while ((timeManager.getRemainingTimeForState(12) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(12));
    Serial.println("Crossing Gap");
    rc.motorDriver.setSpeed(150, 150);
    for (int i = 0; i < 1000; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }
    rc.motorDriver.setSpeed(50, 50);
    for (int i = 0; i < 7000; i++) {
      rc.motorDriver.startMove();
      rc.myservo.write(93);
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }
    rc.motorDriver.setSpeed(-50, -50);
    for (int i = 0; i < 800; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }
    rc.motorDriver.setSpeed(0, 0);
    for (int i = 0; i < 500; i++) {
      rc.motorDriver.startMove();
      rc.myservo.write(120);
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    rc.motorDriver.setSpeed(255, 255);
    for (int i = 0; i < 1000; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }

    rc.motorDriver.setSpeed(0, 0);
    for (int i = 0; i < 1000; i++) {
      rc.motorDriver.startMove();
      vTaskDelay(1 / portTICK_PERIOD_MS);
    }
    break;
  }
  currentState = FOLLOW_LINE;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(12, stateComplete);
  xSemaphoreGive(bufferMutex);
}

// display_logo
// State Number 5
// Current Max Time 2 seconds
void display_logo() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  Serial2.println("DISPLAY_LOGO");
  while ((timeManager.getRemainingTimeForState(15) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(15));
    Serial.println("Displaying Logo");
    //rc.doTheJig();
  }
  currentState = PUSH_BUTTON;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(15, stateComplete);
  xSemaphoreGive(bufferMutex);
}


// push_button
// State Number 16
// Current Max Time 3 seconds
void push_button() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  Serial2.println("PUSH_BUTTON");
  while ((timeManager.getRemainingTimeForState(16) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(16));
    Serial.println("Pushing Button");
    // if (rc.circleRight()) {
    //   if (rc.circleLeft()) {
    //     if (timeManager.getRemainingTimeForState(1) < 1500) {
    //       if (rc.stopTheMotors()) {
    //         stateComplete = true;
    //         break;
    //       }
    //     }
    //   }
    // }
  }
  currentState = DONE;
  timeManager.endState(16, false);
  if (debugMode) {
    vTaskSuspend(debugTaskHandle);
  }
}

// done
// State Number 17
// Current Max Time 0 seconds (Doesn't have time limit)
void done() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  Serial2.println("DONE");
  Serial.println("Done");
  long rTime = timeManager.getRemainingTime();
  long runTime = timeManager.getRunTime();
  Serial.print("Run Time: ");
  Serial.println(runTime);
  Serial.print("Remaining Time: ");
  Serial.println(rTime);
  vTaskSuspend(readDetTaskHandle);
  vTaskSuspend(processDetTaskHandle);
  vTaskSuspend(SensorBoxTaskHandle);
  if (debugMode) {
    vTaskSuspend(debugTaskHandle);
  }
  vTaskDelay(1500 / portTICK_PERIOD_MS);
  // currentState =  WAIT_FOR_START;
  // Follow_Line_Counter = 0;
  // vTaskResume( readDetTaskHandle );
  // vTaskResume( processDetTaskHandle );
  // vTaskResume( SensorBoxTaskHandle );
  // if (debugMode){
  //   vTaskResume( debugTaskHandle );
  // }
  if (rc.stopTheMotors()) {
  }
  timeManager.endState(17, stateComplete);
}

// emergency_stop
// State Number 18
// Current Max Time 0 seconds (Doesn't have time limit)
void emergency_stop() {
  bool stateComplete = false;
  // vTaskDelay(100 / portTICK_PERIOD_MS);
  Serial2.println("EMERGENCY_STOP");
  Serial.println("EMERGENCY STOP");
  vTaskDelay(pdMS_TO_TICKS(1500 + random(4000)));  // Delay for 1 to 5 seconds
  currentState = DONE;
  timeManager.endState(18, stateComplete);
}


void line_follow(int stateId, RobotState nextState, int Follow_Line_Counter, bool stateComplete) {
  int Speed = 150;
  while ((timeManager.getRemainingTimeForState(stateId) > 0) && !stateComplete && !timeManager.timeOut) {
    Serial.println(timeManager.getRemainingTimeForState(stateId));
    Serial.print("Following Line.");
    Serial.println(Follow_Line_Counter);
    while (abs(rc.GetPixyAngle()) < 40) {
      rc.lineFollow(Speed, 0);
    }

    rc.motorDriver.setSpeed(150, 150);
    while (rc.USDistance() > 23) {
      rc.motorDriver.startMove();
      delay(1);
    }
    break;
  }
  currentState = nextState;
  xSemaphoreTake(bufferMutex, portMAX_DELAY);
  timeManager.endState(stateId, stateComplete);
  xSemaphoreGive(bufferMutex);
}
