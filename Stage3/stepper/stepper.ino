#include <math.h>
#include <Stepper.h>
#define STEPS 2048  //定義步進馬達每圈的步數

//steps:代表馬達轉完一圈需要多少步數。如果馬達上有標示每步的度數，
//將360除以這個角度，就可以得到所需要的步數(例如：360/3.6=100)。(int)

const float PIF = acos(-1);
const float SCALE = STEPS / 2 / PIF;

Stepper pitch(STEPS, 13, 11, 12, 10);
Stepper yaw(STEPS, 5, 3, 4, 2);

// float datatype
union Float
{
  char bytes[4];
  float val;
};

void readFloat(Float* x)
{
  while (Serial.available() < 4);
  for (int i = 0; i < 4; ++i) {
    x->bytes[i] = Serial.read();
  }
}

inline void rotate(Stepper* x, float rad)
{
  x->step((int)(rad * SCALE));
}

Float p, y;

void setup()
{
  Serial.begin(115200);
  pitch.setSpeed(5);     // 將馬達的速度設定成140RPM 最大  150~160
  yaw.setSpeed(5);       // 將馬達的速度設定成140RPM 最大  150~160
  digitalWrite(6, HIGH);
}

void loop()
{
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    Serial.println("-------------------");
    if (cmd == 'p') {
      Serial.println("change pitch:");
      readFloat(&p);
      rotate(&pitch, p.val);
      Serial.println(p.val);
    }
    else if (cmd == 'y') {
      Serial.println("change yaw:");
      readFloat(&y);
      rotate(&yaw, y.val);
      Serial.println(y.val);
    }
  }
}
