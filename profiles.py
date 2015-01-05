import serial
ser = serial.Serial('/dev/ttyACM0', 9600)

def action(name):
    if name == "master":
        ser.write(str(255).encode())
