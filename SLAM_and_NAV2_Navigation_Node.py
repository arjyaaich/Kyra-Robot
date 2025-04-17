SLAM_and_NAV2_Navigation_Node


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