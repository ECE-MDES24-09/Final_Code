//
// Created by jordan on 01/31/2024.
//

// DetectionBuffer.h - Header file for managing detections and their storage

#include <string.h>

#ifndef DETECTION_BUFFER_H
#define DETECTION_BUFFER_H

// Structure to represent a gyro data
struct JetsonGyro {

    float x, y, z; // 3D coordinates of the detected object
   

    // Constructor to initialize the JetsonGyro structure
    JetsonGyro(float _x = 0.0f,
              float _y = 0.0f,
			  float _z = 0.0f,
              )
    {
        x = _x;
		y = _y;
		z = _z;
        
    }

    // Assignment operator
    JetsonGyro& operator=(const JetsonGyro& other);
};


#endif
