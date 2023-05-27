int ledPin = 32;

void setup()
{
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
}

void loop()
{
  
  while(Serial.available()){
    int x = Serial.read();
    if(x == '1')
    {
      digitalWrite(ledPin, HIGH);
    }
    else if(x == '0')
    {
      digitalWrite(ledPin, LOW);
    }
  }
  delay(10);
}
