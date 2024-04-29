#include "pixy_line_detection.h"


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