#ifndef globalVariables_h
#define globalVariables_h

//Time Management Variables
    // WAIT_FOR_START ID == 0. It does not count toward timer.
    inline long _startTime;
    inline long _maxTime = 105000;
    inline long _stateTimeLimits[19] = { 0, 10000, 10000, 5000, 4000, 1500, 2000, 1500, 5000, 3000, 10000, 5000, 15000, 5000, 15000, 2000, 3000, 0, 0 };
    inline int _numStates = 19;
    inline int _currentState;
    inline long _stateStartTime;
    inline long _stateEndTime;
    inline long _currentstateMaxTime;
    inline long _timeBank;
    inline const long STATE_SWITCH_BUFFER = 500; // 0.5 seconds in milliseconds
    inline bool _isStartStateCompleted;
    inline const int WAIT_FOR_START_STATE_ID = 0;

// Serial Communication Variables
    #define DATA_LENGTH 5
    #define START_WORD "START"
    #define START_WORD_LEN 5

    inline unsigned long lastTimeRecevieData = 0;

////RobotStateController Variables
    #define BIT_NEW_DATA_AVAILABLE (1 << 0)
    #define BIT_READ_DETECTIONS    (1 << 1)

    // Full Run
    //inline constexpr int NUM_STATES = 19;
    // Move Around Board Run
    // inline constexpr int NUM_STATES = 9;
    #define NUM_STATES 9
    inline const int lightSensorPin = A12;
    inline const int threshold = 260;


    // Enum for different event states
    enum Event {
        START_EVENT,
        STOP_EVENT,
        TIMEOUT_EVENT,
    };


    // Enum for different robot states
    enum RobotState {
        WAIT_FOR_START,
        GET_BIG_BOXES,
        GET_SMALL_BOXES,
        DEPOSIT_BIG_BOXES,
        DEPOSIT_SMALL_BOXES,
        FOLLOW_LINE, // Has Counter
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

// Robot Control Variables
    #define MAX_ANGLE 70.0
    #define SPEED_MULTIPLIER 1.0
    #define ROLLER_MULTIPLIER 1.0
    // Counter and compare values
    inline const uint16_t t1_load = 0;
    inline const uint16_t t1_comp = 250;
    inline const int sensorPin = A12;

    enum class RollerState
    {
        InTake,
        OutTake,
        Stop,
    };

    enum class StorageState
    {
        Start,
        PickUp,
        CubeDrop,
        FuelDrop,
        Cruise
    };

    inline int8_t dataRecieved[DATA_LENGTH];

    inline RollerState rollersState = RollerState::Stop;
    inline StorageState storageState = StorageState::Start;

    inline float targetStorageAng = 90.0f;
    inline float storageAng = 0.0f;
    inline float storageRotateSpeed = 0.1f; // degree per loop


    inline int8_t lastDataReceived2 = -1; // To track the last state of dataRecieved[2]
    inline int8_t lastDataReceived3 = -1; // To track the last state of dataRecieved[3]
    inline int8_t lastDataReceived4 = -1; // To track the last state of dataRecieved[4]
    inline int storageCycleStep = 0;

//Manipulator Control Variables
    inline const int STORAGE_SERVO_PIN = 12;
    inline const int LEFT_ROLLER_SERVO_PIN = 10; // Example pin for left roller servo
    inline const int RIGHT_ROLLER_SERVO_PIN = 11; // Example pin for right roller servo

    inline float currentStorageAngle; // Current angle of the storage servo
    inline float targetStorageAngle; // Target angle for the storage servo

//Robot and Manipulater Shared Variables
    inline const float CubeDropAng = 115.0f;
    inline const float pickUpAng = 115.0f;
    inline const float crusingAng = 90.0f;
    inline const float fuelDropAng = 75.0f;

// Gyro blue Variables
    inline const unsigned long microsPerReading = 1000000 / 25;
    inline unsigned long microsPrevious = 0;

//Motor Driver Variables
    // motor pins
    #define RIGHT_FORWARD_PIN 5
    #define RIGHT_BACKWARD_PIN 4
    #define LEFT_FORWARD_PIN 3
    #define LEFT_BACKWARD_PIN 2

    #define RAMP_RATE 2 

    inline int OS = 0;

    inline int leftSpeed = 0;
    inline int rightSpeed = 0;

    inline int targetLeftSpeed = 0;
    inline int targetRightSpeed = 0;

//Pixy Line Detection Variables
    #define CAM_ANG 20  // camera angel relative to the plane where the line at
    #define VER_FOV 40  // vertical fov of camera

    #define HOR_PIXEL 78  // horizontal pixel
    #define VER_PIXEL 51  // vertical pixel

    #define CAM_HEIGHT 200  // height of camera from ground in milimeter

    // member variables initialization
    
    inline double x0 = 0;
    inline double y0 = 0;
    inline double x1 = 0;
    inline double y1 = 0;

//TMP Variables
    enum SerialSelect {
        S0,
        S1,
        S2,
        S3
    };

    inline int16_t tfDist;    // Distance to object in centimeters
    inline int16_t tfFlux;    // Strength or quality of return signal
    inline int16_t tfTemp;    // Internal temperature of Lidar sensor chip
#endif
