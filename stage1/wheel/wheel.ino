#include <AccelStepper.h>

#define W1_CLK 9
#define W1_DIR 8
#define W2_CLK 7
#define W2_DIR 6
#define W3_CLK 5
#define W3_DIR 4
#define W4_CLK 3
#define W4_DIR 2

// stepper objects
AccelStepper w[4] = {
  AccelStepper(AccelStepper::DRIVER, W1_CLK, W1_DIR),
  AccelStepper(AccelStepper::DRIVER, W2_CLK, W2_DIR),
  AccelStepper(AccelStepper::DRIVER, W3_CLK, W3_DIR),
  AccelStepper(AccelStepper::DRIVER, W4_CLK, W4_DIR)
};

// CONSTANTS
float MAX_OMEGA = 200.0;
float VC = 150;
float OC = 3;
float R_W = 5;
float HR = 8.75 + 16;
float omega[4];

void set_omega(float vx, float vy, float om) {
  omega[0] = (vy - vx + om * HR) / R_W;
  omega[1] = (vy + vx - om * HR) / R_W;
  omega[2] = (vy + vx + om * HR) / R_W;
  omega[3] = (vy - vx - om * HR) / R_W;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  for (int i = 0; i < 4; ++i) {
    w[i].setMaxSpeed(MAX_OMEGA);
    w[i].setSpeed(0.0);
    omega[i] = 0;
  }
  /*w[0].setSpeed(50.0);*/
}

void loop() {
  // put your main code here, to run repeatedly:
  /*if (Serial.available() > 0) {*/
    /*char cmd = Serial.read();*/

    /*if (cmd == 'w') {*/
      /*set_omega(0, VC, 0);*/
    /*} else if (cmd == 's') {*/
      /*set_omega(0, -VC, 0);*/
    /*} else if (cmd == 'a') {*/
      /*set_omega(-VC, 0, 0);*/
    /*} else if (cmd == 'd') {*/
      /*set_omega(VC, 0, 0);*/
    /*} else if (cmd == 'q') {*/
      /*set_omega(0, 0, OC);*/
    /*} else if (cmd == 'r') {*/
      /*set_omega(0, 0, -OC);*/
    /*} else {*/
      /*set_omega(0, 0, 0);*/
    /*}*/

    /*Serial.println("-------------------");*/
    /*Serial.println(cmd);*/
  /*}*/
  /*else {*/
    /*set_omega(0, 0, 0);*/
  /*}*/

  for (int i = 0; i < 4; ++i) {
    /*w[i].setSpeed(omega[i]);*/
    w[i].runSpeed();
  }
}
