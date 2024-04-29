#ifndef TMP_H
#define TMP_H

#include <TFMPlus.h>
#include "printf.h"

enum SerialSelect {
	S0,
	S1,
	S2,
	S3
};

class TMP {
public:
	void init(SerialSelect SS);
	int16_t getDist();

private:
	TFMPlus tfmP;
	int16_t tfDist;    // Distance to object in centimeters
	int16_t tfFlux;    // Strength or quality of return signal
	int16_t tfTemp;    // Internal temperature of Lidar sensor chip
};

#endif // TMP_H
