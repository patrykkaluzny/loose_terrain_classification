#include <Wire.h> 
#include <LiquidCrystal_I2C.h>


const int analogInPin = A0;           // Analog input pin that the sensor output is attached to (white wire)
int minADC = 15;                       // replace with min ADC value read in air
int maxADC = 644;                     // replace with max ADC value read fully submerged in water
bool is1V1Output = false; // set true if 1.1V output sensor is used for 3V set to false

LiquidCrystal_I2C lcd(0x27, 16, 2);
 
 
int moistureValue, mappedValue;
 
void setup() {
// initialize serial communications at 9600 bps:
Serial.begin(9600);
 
if(is1V1Output == true)
  analogReference(INTERNAL); //set ADC reference to internal 1.1V

// initialize the LCD
lcd.begin();

// Turn on the blacklight and print a message.
lcd.backlight();
}
 
void loop() {
// read the moisture value:
moistureValue = analogRead(analogInPin);
 
// print ADC results to the serial monitor:
// Serial.print("ADC = " );
Serial.print(moistureValue);
//Serial.print(", " );
 
mappedValue = map(moistureValue,minADC,maxADC, 0, 100); 
 
// print mapped results to the serial monitor:
lcd.clear();
Serial.print("Moisture value = " );
Serial.println(mappedValue);
lcd.setCursor(0,0);
lcd.print("Moisture value = ");
lcd.setCursor(0,1);
lcd.print(mappedValue);
lcd.setCursor(3,1);
lcd.print("%");
// wait 500 milliseconds before the next loop
delay(1000);
}
