Kyra Robot full code with circuits




import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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




import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import tf_transformations

class NavigationSystem(Node):
    def __init__(self):
        super().__init__('navigation_system')
        self.navigator = BasicNavigator()
        
       
        self.slam = self.create_subscription(
            LaserScan,
            '/scan',
            self.slam_callback,
            10)
        
        
        self.initialize_map()
    
    def initialize_map(self):
       
        self.navigator.waitUntilNav2Active()
        
       
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.position.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        initial_pose.pose.orientation.x = q[0]
        initial_pose.pose.orientation.y = q[1]
        initial_pose.pose.orientation.z = q[2]
        initial_pose.pose.orientation.w = q[3]
        
        self.navigator.setInitialPose(initial_pose)
    
    def slam_callback(self, msg):
        
        pass
    
    def navigate_to_pose(self, x, y, theta):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, theta)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]
        
        self.navigator.goToPose(goal_pose)

def main(args=None):
    rclpy.init(args=args)
    navigation = NavigationSystem()
    rclpy.spin(navigation)
    navigation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


    


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


    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()





    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    Main Control Node Python


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import openai
import speech_recognition as sr
from gtts import gTTS
import pygame
import mediapipe as mp
import torch
from transformers import pipeline
from nav2_simple_commander.robot_navigator import BasicNavigator
from ament_index_python.packages import get_package_share_directory

class KyraRobot(Node):
    def __init__(self):
        super().__init__('kyra_robot')
        
        
        openai.api_key = 'your-api-key'
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_control_pub = self.create_publisher(String, '/arm_control', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
       
        self.navigator = BasicNavigator()
        
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
       
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        
        self.sign_language_model = pipeline('image-classification', model='sign-language-model')
        
      
        self.wake_word = "kyra stalnaker"
        self.is_awake = False
        
     
        self.create_timer(0.1, self.main_loop)
    
    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
       
        results = self.face_detection.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                
                pass
        
        
        hand_results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
            
            pass
    
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, phrase_time_limit=3)
        
        try:
            text = self.recognizer.recognize_google(audio).lower()
            if self.wake_word in text:
                self.is_awake = True
                self.speak("Hello, how can I help you today?")
        except sr.UnknownValueError:
            pass
    
    def process_voice_command(self, command):
       
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a home assistant robot named Kyra. Respond concisely."},
                {"role": "user", "content": command}
            ]
        )
        
        reply = response.choices[0].message.content
        self.speak(reply)
        
        self.execute_command(command, reply)
    
    def execute_command(self, command, ai_response):
       
        if "go to" in command.lower():
            room = self.extract_room(command)
            self.navigate_to(room)
        
        
        elif "clean" in command.lower():
            if "dishes" in command.lower():
                self.clean_dishes()
            elif "laundry" in command.lower():
                self.do_laundry()
            else:
                self.clean_house()
        
        
        elif "pick up" in command.lower() or "grab" in command.lower():
            obj = self.extract_object(command)
            self.pick_and_place(obj)
    
    def navigate_to(self, room):
        
        waypoints = {
            "kitchen": [1.0, 2.0, 0.0, 1.0],
            "living room": [3.0, 1.5, 0.0, 1.0],
            "bedroom": [2.0, 3.5, 0.0, 1.0]
        }
        
        if room in waypoints:
            self.navigator.goToPose(waypoints[room])
            self.speak(f"Going to the {room}")
    
    def clean_dishes(self):
        
        self.control_arm("move_to_sink")
        self.control_arm("pick_sponge")
        self.control_arm("scrub_dish")
       
    
    def do_laundry(self):
        
        self.navigate_to("laundry room")
        self.control_arm("pick_clothes")
        self.control_arm("load_washer")
       
    
    def pick_and_place(self, obj):
        
        self.speak(f"Looking for {obj}")
       
        self.control_arm(f"pick_{obj}")
        self.control_arm(f"place_{obj}")
    
    def control_arm(self, command):
        
        msg = String()
        msg.data = command
        self.arm_control_pub.publish(msg)
    
    def speak(self, text):
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    def main_loop(self):
        if not self.is_awake:
            self.listen_for_wake_word()
        else:
            
            pass

def main(args=None):
    rclpy.init(args=args)
    kyra = KyraRobot()
    rclpy.spin(kyra)
    kyra.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()