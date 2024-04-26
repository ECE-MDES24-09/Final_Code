#include <robot_control_1.h>
#include <Wire.h>

#define MOTOR_SPEED 125

RobotControl rc;
float angle = 0.0f;
int Offset = 30;

void PickUpCubes();
void FollowLine();
void TurnLeft90();
void CrossGap();
  
void setup() { 
  Serial.begin(115200);
  Serial1.begin(115200);
  rc.init();
}

void loop() {
    //Delays start untill light turns on
    //for(int i = 0; i<50; i++){
    //  rc.RampInfeed(Offset+120);
      
    //}
    int Testing = 1;
    while(Testing<1){
    //Serial.println(rc.USDistance());
    Serial.println(rc.GetPixyAngle());
    //analogWrite(5,255);
		//analogWrite(3,255);
    //rc.test(); 
    }

     while(rc.ColorSensor()>100){

     Serial.println(rc.ColorSensor());
     //Serial.println(rc.USDistance());
    }

    for(int i = 0; i<50; i++){
      rc.RampInfeed(Offset+120);
    }
    //PickUpCubes();

    rc.motorDriver.setSpeed(-150,150);
		for(int i = 0; i<1700; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    FollowLine();


    //Spit out cubes
    //rc.motorDriver.Infeed(-250);
    //delay(2000);


    TurnLeft90();
    FollowLine();
    //PickUpThrusters();
    TurnLeft90();

  //DO NOT use linefollow function, you must use the code below.
    for(int i = 0; i<37;i++){
		rc.RampInfeed(Offset+115);
		}
  while(abs(rc.GetPixyAngle())<40){
  rc.lineFollow(200, 0);
  }

    CrossGap();
    
    for(int i = 0; i<37;i++){
		rc.RampInfeed(Offset+120);
		}
    //just presses button, need to code thruster drop off.
    rc.motorDriver.setSpeed(100,100);
    while(rc.USDistance()>20){
		rc.motorDriver.startMove();
    delay(1);
    }

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}

  	rc.motorDriver.setSpeed(-150,150);
		for(int i = 0; i<1400; i++){
		rc.motorDriver.startMove();
    delay(1);
		}

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    rc.FlagWave(90);
    delay(1);
		}

    for(int i = 0; i<37;i++){
		rc.RampInfeed(Offset+120);
		}

    rc.motorDriver.setSpeed(-100,-100);
    while(rc.USDistance()<36){
		rc.motorDriver.startMove();
    delay(1);
    }

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}

    rc.motorDriver.setSpeed(100,100);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    //Stops the robot
  int STOP = 0;
  while(STOP<1){}
}

void PickUpCubes(){

  	for(int i = 0; i<37;i++){
		rc.RampInfeed(Offset+125);
		}

  rc.motorDriver.Infeed(1);

  delay(100);
  for(int i = 0; i<2; i++){
    rc.motorDriver.setSpeed(50,50);
    //Serial.println(rc.USDistance());
    while(rc.USDistance()>11){
		rc.motorDriver.startMove();
    Serial.println(rc.USDistance());
    delay(1);
    }

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}

    rc.motorDriver.setSpeed(-50,-250);
		for(int i = 0; i<800; i++){
		rc.motorDriver.startMove();
    delay(1);
    }

    rc.motorDriver.setSpeed(-250,-50);
		for(int i = 0; i<800; i++){
		rc.motorDriver.startMove();
    delay(1);
    }

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
  }

    rc.motorDriver.setSpeed(50,50);
    //Serial.println(rc.USDistance());
    while(rc.USDistance()>11){
		rc.motorDriver.startMove();
    Serial.println(rc.USDistance());
    delay(1);
    }

    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
    rc.RampInfeed(Offset+115);
		}

    rc.motorDriver.setSpeed(-200,-200);
		for(int i = 0; i<500; i++){
		rc.motorDriver.startMove();
    delay(1);
    }

    rc.motorDriver.Infeed(0);
    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
}

void FollowLine(){

    for(int i = 0; i<50; i++){
      rc.RampInfeed(Offset+110);
    } 

    while(abs(rc.GetPixyAngle())<40){
    rc.lineFollow(150, 0);
    }

    for(int i = 0; i<50; i++){
      rc.RampInfeed(Offset+125);
    }
    delay(1000);
    rc.motorDriver.setSpeed(150, 150);
    while(rc.USDistance()>23){
		rc.motorDriver.startMove();
    delay(1);
    }

    for(int i = 0; i<50; i++){
      rc.RampInfeed(Offset+120);
    }

    /*rc.motorDriver.setSpeed(0, 0);
	  for(int i = 255;i>0;i--){
		rc.motorDriver.startMove();
    delay(1);
	}*/

}

void TurnLeft90(){
  
    rc.motorDriver.setSpeed(-150,150);
		for(int i = 0; i<1400; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
}

void CrossGap(){

  	rc.motorDriver.setSpeed(150,150);
		for(int i = 0; i<1000; i++){
		rc.motorDriver.startMove();
    rc.RampInfeed(Offset+88);
    delay(1);
		}
  	rc.motorDriver.setSpeed(60,60);
		for(int i = 0; i<4500; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    rc.motorDriver.setSpeed(-50,-50);
		for(int i = 0; i<1200; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    rc.RampInfeed(Offset+121);
    delay(1);
		}
  	rc.motorDriver.setSpeed(255,255);
		for(int i = 0; i<1000; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
    rc.motorDriver.setSpeed(0,0);
		for(int i = 0; i<255; i++){
		rc.motorDriver.startMove();
    delay(1);
		}
}

void pickUpClostest() {
  rc.lineDetect.update();
    DisAng scDisAng = rc.lineDetect.getNearestSmallCubeDisAng();
  if (scDisAng.distance == -1){
    rc.motorDriver.setSpeed(0, 0);
  }
  else{

    int speedOffset = 10 * (100 * (scDisAng.angle) / 90);

    int leftSpeed = 100 + speedOffset;
    int rightSpeed = 100 - speedOffset;

    if (leftSpeed > 255)
    leftSpeed = 255;
    else if (leftSpeed < -255)
      leftSpeed = -255;

    if (rightSpeed > 255)
      rightSpeed = 255;
    else if (rightSpeed < -255)
    rightSpeed = -255;

    rc.motorDriver.setSpeed(leftSpeed, rightSpeed);
  }

  rc.motorDriver.startMove();

}