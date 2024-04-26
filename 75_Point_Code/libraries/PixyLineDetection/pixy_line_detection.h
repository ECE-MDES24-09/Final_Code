#ifndef PIXY_LINE_DETECTION_H
#define PIXY_LINE_DETECTION_H

#define CAM_ANG 45  // camera angel relative to the plane where the line at
#define VER_FOV 40  // vertical fov of camera

#define HOR_PIXEL 78  // horizontal pixel
#define VER_PIXEL 51  // vertical pixel

#define CAM_HEIGHT 200  // height of camera from ground in milimeter

#include <Pixy2.h>
#include "Arduino.h"

struct xy {
	double x;
	double y;
};

class PixyLineDetect {

public:
	void init();
	
	// refresh the line vector
	// A low-pass filter may be applied in the future to reduce noise
	void update();
	
	// return the angle between pixy camera and the line in degree
	// positive: line tilts to the right of camera
	// negative: line tilts to the left of camera
	double getAng() const;
	double getAng(double maxAng) const;

	// return distance of the camera to the line in milimeter
	// return negative number if camera is at left of the line, return postive numebr if camera is at right of the line
	double getOffset() const;
	double getOffset(double maxAng) const;

private:
	
	// This function takes xy coordiate from the camera view to line plane coordiate. The camera is at the origin. The unit is milimeter
	xy findHorXY(double x, double y) const;

	// member variables initialization
	Pixy2 pixy;
	double x0 = 0;
	double y0 = 0;
	double x1 = 0;
	double y1 = 0;

};

void PixyLineDetect::init() {
	pixy.init();
	Serial.println(pixy.changeProg("line"));
}

void PixyLineDetect::update() {
	pixy.line.getMainFeatures();
	if (pixy.line.numVectors) {
		x0 = (double)pixy.line.vectors->m_x0;
		y0 = (double)pixy.line.vectors->m_y0;
		x1 = (double)pixy.line.vectors->m_x1;
		y1 = (double)pixy.line.vectors->m_y1;
	}
}

double PixyLineDetect::getAng() const {
	xy xyout1 = findHorXY(x0, y0);
	xy xyout2 = findHorXY(x1, y1);

	double xo1 = xyout1.x;
	double yo1 = xyout1.y;

	double xo2 = xyout2.x;
	double yo2 = xyout2.y;

	if (yo1 == yo2) {
		if (xo1 >= xo2)
			return -M_PI / 2.0 / M_PI * 180.0;
		if (xo1 <= xo2)
			return M_PI / 2.0 / M_PI * 180.0;

	}

	return atan((xo1 - xo2) / (yo1 - yo2)) / M_PI * 180.0;
}

double PixyLineDetect::getAng(double maxAng) const {
	double ang = getAng();

	if (ang > maxAng)
		ang = ang - 180.0;

	return ang;
}

double PixyLineDetect::getOffset() const {

	double ang_rad = getAng() / 180.0 * M_PI;
	xy xyo = findHorXY(x0, y0);
	double xo0 = xyo.x * cos(ang_rad) - xyo.y * sin(ang_rad);

	return -xo0;
}

double PixyLineDetect::getOffset(double maxAng) const {
	double ang_rad = getAng(maxAng) / 180.0 * M_PI;
	xy xyo = findHorXY(x0, y0);
	double xo0 = xyo.x * cos(ang_rad) - xyo.y * sin(ang_rad);

	return -xo0;
}

xy PixyLineDetect::findHorXY(double x, double y) const {

	double x_range = HOR_PIXEL - 1.0;
	double y_range = VER_PIXEL - 1.0;

	double alpha = VER_FOV / 2 / 180.0 * M_PI;
	double theta = CAM_ANG / 180.0 * M_PI;

	x = x - 1.0;
	x = x - x_range / 2.0;
	x = x / x_range;
	y = y - 1.0;
	y = y_range - y;
	y = y / y_range;

	double
		k1 = sin(2.0 * alpha) / (cos(alpha + theta) * cos(alpha - theta)),
		k2 = tan(theta - alpha),
		k3 = 2.0 * sin(alpha) / cos(theta - alpha),
		k4 = 1.0 / tan(theta) + tan(theta - alpha),
		k5 = 1.0 / tan(theta);

	double yo = y * CAM_HEIGHT * k1 + CAM_HEIGHT * k2;
	double xo = x * (k3 * x_range) / (k4 * y_range) * (yo + CAM_HEIGHT * k5);

	xy xyout;
	xyout.x = xo;
	xyout.y = yo;

	return xyout;

}

#endif