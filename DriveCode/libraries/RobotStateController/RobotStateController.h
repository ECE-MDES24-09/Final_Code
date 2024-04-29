//
// Created by jorda on 03/15/2024.
//

#ifndef ROBOTSTATECONTROLLER_H
#define ROBOTSTATECONTROLLER_H

#include <Arduino.h>
#include <Arduino_FreeRTOS.h>
#include <semphr.h>
#include <event_groups.h>
#include "robot_control.h"


#define BIT_NEW_DATA_AVAILABLE (1 << 0)
#define BIT_READ_DETECTIONS    (1 << 1)
// Include other necessary headers...
class RobotStateController;
// Full Run
constexpr int NUM_STATES = 19;
// Move Around Board Run
//constexpr int NUM_STATES = 9;
const int lightSensorPin = A0;
const int threshold = 160;


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


// Function pointer type for state handling methods in RobotStateController
typedef void (RobotStateController::*StateFunc)();

// Function pointer type for event handling methods in RobotStateController
typedef void (RobotStateController::*EventFunc)(Event);

// Structure representing a state of the robot
struct State {

    RobotState currState; // Current state of the robot
    int followLineCounter; // Counter for line following logic
    StateFunc func; // Function to call for the current state
    State* prev; // Pointer to previous state
    State* next; // Pointer to next state
    bool stateComplete = false; // Flag to indicate if state is complete
    bool timeOut = false; // Flag to indicate if state has timed out

    // Constructor for State structure
    State(RobotState state, int follow_line_counter, StateFunc func, State *prev, State *next);

    // Destructor for State structure
    ~State();

    // Equality operator overloads for State comparison
    friend bool operator==(const State &lhs, const State &rhs);
    friend bool operator!=(const State &lhs, const State &rhs);

    // Setters and getters for State properties
    void set_follow_line_counter(int follow_line_counter);
    RobotState getCurrentState() const;
    StateFunc getFunc() const;
    void setFunc(StateFunc newFunc);
    State * getPrev() const;
    void setPrev(State *newPrev);
    State * getNext() const;
    void setNext(State *newNext);
};


class RobotStateController {
private:

    State* robotCurrentState;
    State* robotPrevState;
    State* robotNextState;
    MotorDriver motorDriver1;
    // Motor Mutex
    SemaphoreHandle_t motorMutex;
    // States for Full Run
    State states[NUM_STATES] = {
        {State(WAIT_FOR_START, 0, &RobotStateController::wait_for_start, nullptr, &states[1])}, // 0
        {State(GET_BIG_BOXES, 0, &RobotStateController::get_big_boxes, &states[0], &states[2])}, // 1
        {State(GET_SMALL_BOXES, 0, &RobotStateController::get_small_boxes, &states[1], &states[3])}, // 2
        {State(FOLLOW_LINE, 0, &RobotStateController::follow_line, &states[2], &states[4])}, // 3
        {State(GO_TO_RED_ZONE, 1, &RobotStateController::go_to_red_zone, &states[3], &states[5])}, // 4
        {State(DEPOSIT_SMALL_BOXES, 0, &RobotStateController::deposit_small_boxes, &states[4], &states[6])}, // 5
        {State(GO_TO_BLUE_ZONE, 1, &RobotStateController::go_to_blue_zone, &states[5], &states[7])}, // 6
        {State(DEPOSIT_BIG_BOXES, 0, &RobotStateController::deposit_big_boxes, &states[6], &states[8])}, // 7
        {State(FOLLOW_LINE, 1, &RobotStateController::follow_line, &states[7], &states[9])}, // 8
        {State(GO_TO_GREEN_ZONE, 1, &RobotStateController::go_to_green_zone, &states[8], &states[10])}, // 9
        {State(GET_ROCKETS, 1, &RobotStateController::get_rockets, &states[9], &states[11])}, // 10
        {State(FOLLOW_LINE, 2, &RobotStateController::follow_line, &states[10], &states[12])}, // 11
        {State(CROSS_GAP, 2, &RobotStateController::cross_gap, &states[11], &states[13])}, // 12
        {State(FOLLOW_LINE, 3, &RobotStateController::follow_line, &states[12], &states[14])}, // 13
        {State(DEPOSIT_ROCKETS, 3, &RobotStateController::deposit_rockets, &states[13], &states[15])}, // 14
        {State(DISPLAY_LOGO, 3, &RobotStateController::display_logo, &states[14], &states[16])}, // 15
        {State(PUSH_BUTTON, 3, &RobotStateController::push_button, &states[15], &states[17])}, // 16
        {State(DONE, 3, &RobotStateController::done, &states[16], &states[0])}, // 17
        {State(EMERGENCY_STOP, 0, &RobotStateController::emergency_stop, nullptr, nullptr)}, // 18
    };

    /**
    // Move around Board Run
    State states[NUM_STATES] = {
        {State(WAIT_FOR_START, 0, &RobotStateController::wait_for_start, nullptr, &states[1])}, // 0
        {State(FOLLOW_LINE, 0, &RobotStateController::follow_line, &states[0], &states[2])}, // 1
        //{State(GO_TO_RED_ZONE, 1, &RobotStateController::go_to_red_zone, &states[3], &states[5])}, // 4
        //{State(GO_TO_BLUE_ZONE, 1, &RobotStateController::go_to_blue_zone, &states[5], &states[7])}, // 6
        {State(FOLLOW_LINE, 1, &RobotStateController::follow_line, &states[1], &states[3])}, // 2
        //{State(GO_TO_GREEN_ZONE, 1, &RobotStateController::go_to_green_zone, &states[8], &states[10])}, // 9
        {State(FOLLOW_LINE, 2, &RobotStateController::follow_line, &states[2], &states[4])}, // 3
        {State(CROSS_GAP, 2, &RobotStateController::cross_gap, &states[3], &states[5])}, // 4
        {State(FOLLOW_LINE, 3, &RobotStateController::follow_line, &states[4], &states[6])}, // 5
        {State(PUSH_BUTTON, 3, &RobotStateController::push_button, &states[5], &states[7])}, // 6
        {State(DONE, 3, &RobotStateController::done, &states[6], &states[0])}, // 7
        {State(EMERGENCY_STOP, 0, &RobotStateController::emergency_stop, nullptr, nullptr)}, // 8
    };
    **/

    // State handling methods
    void wait_for_start();
    void get_big_boxes();
    void get_small_boxes();
    void follow_line();
    void deposit_big_boxes();
    void deposit_small_boxes();
    void go_to_red_zone();
    void go_to_blue_zone();
    void go_to_green_zone();
    void get_rockets();
    void deposit_rockets();
    void cross_gap();
    void display_logo();
    void push_button();
    void done();
    void emergency_stop();
    static const char* getStateName(RobotState state);

public:
    RobotStateController();
    ~RobotStateController();
    void update();
    RobotState getCurrentRobotState();
    void setState(State* newState);
    State* getPrevState();
    State* getCurrentState();
    State* getNextState();
    State* getState(RobotState searchState, int lineFollowCounter);
    int getStateNum(RobotState searchState, int lineFollowCounter);
    void setEmergencyState();
    int getLineFollowCounter();
    void proceed();
    void goBack();
    void reset();
	void init();
    TaskHandle_t readDetTaskHandle;
    TaskHandle_t processDetTaskHandle;
    EventGroupHandle_t xDetectionsEventGroup;
    RobotControl rc;

};






#endif //ROBOTSTATECONTROLLER_H
