//
// Created by jordan on 01/31/2024.
//

// JetsonGyro.cpp - Implementation of functions for managing JetsonGyro Data

#include "JetsonGyro.h"
#include <string.h>


// Overloaded assignment operator for JetsonGyro structure
JetsonGyro& JetsonGyro::operator=(const JetsonGyro& other) {
    if (this != &other) { // Check for self-assignment
        // Copy data from 'other' JetsonGyro to this JetsonGyro
        
        this->x = other.x;
		this->y = other.y;
		this->z = other.z;
        
    }
    return *this; // Return reference to this object
}
