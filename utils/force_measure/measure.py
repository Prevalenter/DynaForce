# load the pyserial
import serial
import time
import numpy as np

class ForceSensor:
    def __init__(self, port='/dev/ttyUSB0', baudrate=2400, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout, parity=serial.PARITY_NONE)
        self.ser.flush()
        
        self.data_buf = ''
        
        self.data_scale = 0.001
        
        
        self.values = []
        
    def read_force(self):
        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').rstrip()
                try:
                    force = np.array(line.split(','), dtype=float)
                    return force
                except ValueError:
                    pass
            time.sleep(0.01)
            
    def start(self):
        self.ser.write(b'\x65')
        print('start the force sensor')

    
    def stop(self):
        self.ser.write(b'\x81')
        print('stop the force sensor')



    def read_buf_data(self):
        if self.ser.in_waiting > 0:
            # read in hex mode
            data = self.ser.read(self.ser.in_waiting).hex()
            self.data_buf += data
            return self.process_data_buf()



    def process_data_buf(self):
        # split the data_buf by 55
        data_buf_split = self.data_buf.split('55')
        
        # get the last element
        self.data_buf = '55'+data_buf_split[-1]
        
        print(data_buf_split)
        
        data = []
        for data_i in data_buf_split[:-1]:
            rst_i = self.process_data_line(data_i)
            if rst_i is not None:
                # print(rst_i)
                self.values.append(rst_i)
                data.append(rst_i)
                
                # print(self.values)
        return data
    
    def process_data_line(self, line):
    
        if line[:2]=='01' and len(line)==14:
            # print(line, len(line))
            
            if line[2:4]=='2d':
                data_sign = -1
            elif line[2:4]=='2b':
                data_sign = 1
            else:
                print(line)
                raise NotImplementedError()
                
            # to value
            a = int(line[5], 16)
            b = int(line[7], 16)
            c = int(line[9], 16)
            d = int(line[11], 16)
            e = int(line[13], 16)
            
            # breakpoint()
            
            value = data_sign*(a*(16**4) + b*(16**3) + c*(16**2) + d*16 + e)*self.data_scale
            
            return value
            
        elif line[:2]=='02':
            if line[2:4]=='30':
                print('the data is in KG')
            elif line[2:4]=='31':
                print('the data is in 1bf')
            elif line[2:4]=='32':
                print('the data is in NEWTON')
                
            if line[4:6]=='32':
                print('the data scale is:', 0.0001)
            elif line[4:6]=='33':
                print('the data scale is:', 0.001)
            elif line[4:6]=='34':
                print('the data scale is:', 0.01)
            elif line[4:6]=='35':
                print('the data scale is:', 0.1)
            elif line[4:6]=='36':
                print('the data scale is:', 1)   

            return None




if __name__ == '__main__':
    force_sensor = ForceSensor()
    
    force_sensor.start()
    
    for i in range(int(100)):
        print(i)
        force_sensor.read_buf_data()
        
        if i==100:
            force_sensor.stop()
        
        time.sleep(0.1)
    
    # while True:
    #     force = force_sensor.read_force()
    #     print(force)

            