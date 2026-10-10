// 引脚定义（根据实际接线修改）
// DRV8825（1）控制引脚
const int stepPin1 = 13;    // 连接DRV8825（1）的STEP引脚
const int dirPin1 = 12;     // 连接DRV8825（1）的DIR引脚
// DRV8825（2）控制引脚
const int stepPin2 = 11;    // 连接DRV8825（2）的STEP引脚
const int dirPin2 = 10;     // 连接DRV8825（2）的DIR引脚
// DRV8825（3）控制引脚
const int stepPin3 = 9;     // 连接DRV8825（3）的STEP引脚
const int dirPin3 = 8;      // 连接DRV8825（3）的DIR引脚

// 参数定义（可根据需求调整）
const int stepCount = 100;      // 每次转动的步数（修改为100步）
const int stepInterval = 1000;  // 步进间隔（微秒），数值越小转速越快
const int pauseTime = 5000;     // 中间停留时间（毫秒）
const int cyclePause = 20000;   // 循环间停留时间（毫秒）
#define FORWARD HIGH            // 正向（若实际转向相反，改为LOW）
#define BACKWARD LOW            // 反向（若实际转向相反，改为HIGH）

// 函数声明：控制指定电机转动指定方向和步数
void moveMotor(int stepPin, int dirPin, int direction, int steps);

void setup() {
  // 初始化所有控制引脚为输出模式
  pinMode(stepPin1, OUTPUT);
  pinMode(dirPin1, OUTPUT);
  pinMode(stepPin2, OUTPUT);
  pinMode(dirPin2, OUTPUT);
  pinMode(stepPin3, OUTPUT);
  pinMode(dirPin3, OUTPUT);
  
  Serial.begin(9600);  // 初始化串口调试
  Serial.println("系统初始化完成，开始运行...");
}

void loop() {
  // 执行大括号内的运动序列（一个完整循环）
  // 1. DRV8825（3）正转100步 → DRV8825（1）正转100步 → DRV8825（2）正转100步
  Serial.println("阶段1：DRV8825(3)正转100步");
  moveMotor(stepPin3, dirPin3, FORWARD, stepCount);
  Serial.println("阶段1：DRV8825(1)正转100步");
  moveMotor(stepPin1, dirPin1, FORWARD, stepCount);
  Serial.println("阶段1：DRV8825(2)正转100步");
  moveMotor(stepPin2, dirPin2, FORWARD, stepCount);
  delay(pauseTime);  // 延时5秒

  // 2. DRV8825（3）反转100步 → DRV8825（2）反转100步
  Serial.println("阶段2：DRV8825(3)反转100步");
  moveMotor(stepPin3, dirPin3, BACKWARD, stepCount);
  Serial.println("阶段2：DRV8825(2)反转100步");
  moveMotor(stepPin2, dirPin2, BACKWARD, stepCount);
  delay(pauseTime);  // 延时5秒

  // 3. DRV8825（2）正转100步 → DRV8825（3）正转100步
  Serial.println("阶段3：DRV8825(2)正转100步");
  moveMotor(stepPin2, dirPin2, FORWARD, stepCount);
  Serial.println("阶段3：DRV8825(3)正转100步");
  moveMotor(stepPin3, dirPin3, FORWARD, stepCount);
  delay(pauseTime);  // 延时5秒

  // 4. DRV8825（3）反转100步 → DRV8825（2）反转100步
  Serial.println("阶段4：DRV8825(3)反转100步");
  moveMotor(stepPin3, dirPin3, BACKWARD, stepCount);
  Serial.println("阶段4：DRV8825(2)反转100步");
  moveMotor(stepPin2, dirPin2, BACKWARD, stepCount);
  delay(pauseTime);  // 延时5秒

  // 5. DRV8825（2）正转100步 → DRV8825（1）反转100步
  Serial.println("阶段5：DRV8825(2)正转100步");
  moveMotor(stepPin2, dirPin2, FORWARD, stepCount);
  Serial.println("阶段5：DRV8825(1)反转100步");
  moveMotor(stepPin1, dirPin1, BACKWARD, stepCount);
  delay(pauseTime);  // 延时5秒

  // 6. DRV8825（2）正转100步 → DRV8825（3）正转100步
  Serial.println("阶段6：DRV8825(2)正转100步");
  moveMotor(stepPin2, dirPin2, FORWARD, stepCount);
  Serial.println("阶段6：DRV8825(3)正转100步");
  moveMotor(stepPin3, dirPin3, FORWARD, stepCount);
  delay(pauseTime);  // 延时5秒

  // 7. DRV8825（3）反转100步 → DRV8825（2）反转100步 → DRV8825（1）反转100步
  Serial.println("阶段7：DRV8825(3)反转100步");
  moveMotor(stepPin3, dirPin3, BACKWARD, stepCount);
  Serial.println("阶段7：DRV8825(2)反转100步");
  moveMotor(stepPin2, dirPin2, BACKWARD, stepCount);
  Serial.println("阶段7：DRV8825(1)反转100步");
  moveMotor(stepPin1, dirPin1, BACKWARD, stepCount);

  // 循环结束，延时20秒后进入下一次循环
  Serial.println("循环结束，等待20秒后进入下一次循环...");
  delay(cyclePause);
}

// 电机控制函数：参数分别为（步进引脚，方向引脚，方向，步数）
void moveMotor(int stepPin, int dirPin, int direction, int steps) {
  digitalWrite(dirPin, direction);  // 设置方向
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(stepInterval);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(stepInterval);
  }
}