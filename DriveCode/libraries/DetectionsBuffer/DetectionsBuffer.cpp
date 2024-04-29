//
// Created by jordan on 01/31/2024.
//

// DetectionBuffer.cpp - Implementation of functions for managing detections

#include "DetectionsBuffer.h"
#include <string.h>

// Global buffer and index for storing detections
Detection buffer[BUFFER_SIZE];
int bufferIndex = 0;

// Overloaded assignment operator for Detection structure
Detection& Detection::operator=(const Detection& other) {
    if (this != &other) { // Check for self-assignment
        // Copy data from 'other' Detection to this Detection
        strcpy(this->class_name, other.class_name);
        this->timestamp = other.timestamp;
        this->depth_mm = other.depth_mm;
        this->x = other.x;
        this->horizontal_angle = other.horizontal_angle;
        strcpy(this->direction, other.direction);
    }
    return *this; // Return reference to this object
}

// Function to add a detection to the global buffer
void addDetectionToBuffer(const Detection& detection) {
    buffer[bufferIndex] = detection; // Store detection
    bufferIndex = (bufferIndex + 1) % BUFFER_SIZE; // Update index circularly
}

// Function to retrieve a detection from the buffer
Detection getDetectionFromBuffer(int index) {
    if (index < 0 || index >= BUFFER_SIZE) {
        // Return an empty detection if index is out of bounds
        Detection emptyDetection{};
        return emptyDetection;
    }
    return buffer[(bufferIndex + index) % BUFFER_SIZE]; // Retrieve detection
}

// Function to get the current buffer size
int getBufferSize() {
    return BUFFER_SIZE;
}


// Function to get the first detection with the matching name (more for zones)
Detection getSpecificDetection(char name[15]) {
    
    Detection specificDetection;

    // Loop through the buffer to find the first detection with the same name
    for (auto & i : buffer) {
        if( strcmp(name, i.class_name) == 0 )  {
            specificDetection = i;
			break;
        }
    }
    return specificDetection; // Return the detection with the matching name
}



// Function to get the closest detection
Detection getClosestDetection() {
    float minDist = 1000; // Initialize with a large distance value
    Detection closestDetection;

    // Loop through the buffer to find the detection with the smallest distance
    for (auto & i : buffer) {
        if (i.depth_mm > 0 && i.depth_mm < minDist) {
            minDist = i.depth_mm;
            closestDetection = i;
        }
    }
    return closestDetection; // Return the detection with the smallest distance
}


// Function to get the detection with the earliest timestamp
Detection getLatestDetection() {
    float minTime = 10000; // Initialize with a large timestamp value
    Detection latestDetection;

    // Loop through the buffer to find the detection with the earliest timestamp
    for (auto & i : buffer) {
        if (i.timestamp > 0 && i.timestamp < minTime) {
            minTime = i.timestamp;
            latestDetection = i;
        }
    }
    return latestDetection; // Return the detection with the earliest timestamp
}

// Function to get the leftmost detection
Detection getLeftmostDetection() {
    float minAngle = 360; // Initialize with the maximum float value
    Detection leftmostDetection;

    // Loop through the buffer to find the detection with the smallest horizontal angle
    for (auto & i : buffer) {
        if (i.horizontal_angle < minAngle) {
            minAngle = i.horizontal_angle;
            leftmostDetection = i;
        }
    }
    return leftmostDetection; // Return the detection with the smallest horizontal angle
}

void clearBuffer() {
    // Reset each Detection in the buffer
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        buffer[i] = Detection{}; // Assigning a default-constructed Detection
    }
    bufferIndex = 0; // Reset the buffer index to start
}