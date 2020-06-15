# sapereaude

First is necessary run [CyKit](https://github.com/CymatiCorp/CyKit):

- python .\CyKIT.py 127.0.0.1 8765 6 generic+nocounter+noheader+nobattery+ovdelay:300+float

You can change the server and port address but the parameters must be the same

Run application

- python3 sapereaude.py

Actually the demo app is in Spanish

## Raspberry Pi

1. Update Raspberry
- sudo rpi-update
- sudo apt-get update
- sudo apt-get upgrade
- sudo apt-get dist-upgrade

2. Install udev rules
- Create a file with this content and run:  
sudo cp 70-emotiv.rules /etc/udev/rules.d
```
SUBSYSTEM=="hidraw", ACTION=="add",  SUBSYSTEMS=="hid", DRIVERS=="generic-usb", KERNEL=="hidraw*", GROUP="username", MODE="0666"
SUBSYSTEM=="hidraw", ACTION=="add",  SUBSYSTEMS=="usb", DRIVERS=="usbhid", GROUP="username", MODE="0666"
SUBSYSTEM=="usb", ACTION=="add", ATTR{idVendor}=="1234", ATTR{idProduct}=="ed02", GROUP="username", MODE="0666"
SUBSYSTEM=="usb", ACTION=="add", SUBSYSTEMS=="usb", ATTRS{idVendor}=="1d6b", GROUP="username", MODE="0666"
```

3. Restart udev
- sudo service udev restart

4. Required software
- sudo apt-get install git
- sudo apt-get install libglib2.0-dev
- sudo apt-get install libdbus-1-dev
- sudo apt-get install libudev-dev
- sudo apt-get install libical-dev
- sudo apt-get install libreadline-dev
- sudo apt-get install libudev-dev
- sudo apt-get install libusb-1.0.0-dev
- sudo apt-get install libfox-1.6-dev
- sudo apt-get install autotools-dev
- sudo apt-get install automake
- sudo apt-get install libtool
- sudo apt-get install cmake

5. Install Python 3
- sudo apt-get install python3

6. Pip Modules
- Crypto
- PCrypto
- Setuptools
- Tensorflow (1.15)
- Keras
- numpy

7. Architecture
<img src="https://github.com/josuuribe/sapereaude/blob/master/images/NeuralArchitecture.png" width=34% height=34%  />
- (Neural Architecture)[https://github.com/josuuribe/sapereaude/blob/master/images/NeuralArchitecture.png]
- (Workflow)[https://github.com/josuuribe/sapereaude/blob/master/images/Workflow.png]
