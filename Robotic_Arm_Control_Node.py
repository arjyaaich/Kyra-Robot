Robotic_Arm_Control_Node


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from adafruit_servokit import ServoKit

class RoboticArm(Node):
    def __init__(self):
        super().__init__('robotic_arm')
        
        
        self.kit = ServoKit(channels=16)
        
       
        self.positions = {
            "home": [90, 90, 90, 90, 90, 90],
            "pick_sponge": [45, 60, 30, 90, 45, 0],
            
        }
        
        
        self.subscription = self.create_subscription(
            String,
            '/arm_control',
            self.command_callback,
            10)
    
    def command_callback(self, msg):
        command = msg.data
        self.get_logger().info(f'Received command: {command}')
        
        if command in self.positions:
            self.move_to_position(self.positions[command])
        elif command.startswith("pick_"):
            self.execute_pick_sequence(command[5:])
        elif command.startswith("place_"):
            self.execute_place_sequence(command[6:])
    
    def move_to_position(self, angles):
        for i in range(6):
            self.kit.servo[i].angle = angles[i]
            time.sleep(0.1)
    
    def execute_pick_sequence(self, obj):
        
        pass
    
    def execute_place_sequence(self, location):
       
        pass

def main(args=None):
    rclpy.init(args=args)
    robotic_arm = RoboticArm()
    rclpy.spin(robotic_arm)
    robotic_arm.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()