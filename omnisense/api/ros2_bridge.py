"""
ROS2 Bridge for OMNISENSE platform.

Provides ROS2 interface for robotic systems integration.
Publishes tracking data, poses, and system state as ROS2 topics.
"""

import json
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped, PoseArray, Twist, Point, Quaternion
    from std_msgs.msg import Header, String
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker, MarkerArray
    import numpy as np
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy base class

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ROS2Config:
    """ROS2 bridge configuration."""
    node_name: str = "omnisense_bridge"
    namespace: str = "omnisense"
    publish_rate: float = 30.0  # Hz
    qos_reliability: str = "reliable"  # "reliable" or "best_effort"
    qos_history_depth: int = 10


class OmniSenseROS2Node(Node):
    """
    ROS2 node for OMNISENSE bridge.

    Publishes:
    - /omnisense/persons: PoseArray with all tracked persons
    - /omnisense/person/{id}/pose: Individual person poses
    - /omnisense/person/{id}/velocity: Person velocities
    - /omnisense/world_state: JSON string with complete state
    - /omnisense/markers: Visualization markers
    - /omnisense/point_cloud: 3D point cloud
    - /omnisense/alerts: Alert messages
    """

    def __init__(self, config: ROS2Config, app_instance=None):
        """
        Initialize ROS2 node.

        Args:
            config: ROS2 configuration
            app_instance: OMNISENSE application instance
        """
        super().__init__(config.node_name, namespace=config.namespace)

        self.config = config
        self.app_instance = app_instance

        # Setup QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if config.qos_reliability == "reliable"
                       else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=config.qos_history_depth
        )

        # Create publishers
        self.persons_pub = self.create_publisher(
            PoseArray, 'persons', qos_profile
        )

        self.world_state_pub = self.create_publisher(
            String, 'world_state', qos_profile
        )

        self.markers_pub = self.create_publisher(
            MarkerArray, 'markers', qos_profile
        )

        self.point_cloud_pub = self.create_publisher(
            PointCloud2, 'point_cloud', qos_profile
        )

        self.alert_pub = self.create_publisher(
            String, 'alerts', qos_profile
        )

        # Individual person publishers (created dynamically)
        self.person_pose_pubs: Dict[int, Any] = {}
        self.person_velocity_pubs: Dict[int, Any] = {}

        # Create timer for periodic publishing
        timer_period = 1.0 / config.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_callback)

        logger.info(f"ROS2 node '{config.node_name}' initialized at {config.publish_rate} Hz")

    def publish_callback(self):
        """Periodic publishing callback."""
        if not self.app_instance or not self.app_instance.sensor_fusion:
            return

        try:
            # Get world state
            world_state = self.app_instance.sensor_fusion.get_world_state()

            # Publish persons as PoseArray
            self._publish_persons_pose_array(world_state)

            # Publish world state as JSON
            self._publish_world_state_json(world_state)

            # Publish visualization markers
            self._publish_markers(world_state)

            # Publish individual person data
            self._publish_individual_persons(world_state)

        except Exception as e:
            self.get_logger().error(f"Error in publish callback: {e}")

    def _publish_persons_pose_array(self, world_state):
        """Publish all persons as PoseArray."""
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "world"

        for person in world_state.persons.values():
            pose = PoseStamped()

            # Position
            pose.pose.position.x = float(person.position_3d[0])
            pose.pose.position.y = float(person.position_3d[1])
            pose.pose.position.z = float(person.position_3d[2])

            # Orientation (from head pose if available)
            if person.head_pose:
                # Convert Euler angles to quaternion
                quat = self._euler_to_quaternion(
                    person.head_pose.get('roll', 0),
                    person.head_pose.get('pitch', 0),
                    person.head_pose.get('yaw', 0)
                )
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]
            else:
                # Identity quaternion
                pose.pose.orientation.w = 1.0

            pose_array.poses.append(pose.pose)

        self.persons_pub.publish(pose_array)

    def _publish_world_state_json(self, world_state):
        """Publish complete world state as JSON."""
        state_dict = {
            "timestamp": world_state.timestamp,
            "person_count": len(world_state.persons),
            "persons": []
        }

        for person in world_state.persons.values():
            person_dict = {
                "global_id": person.global_id,
                "position_3d": person.position_3d.tolist(),
                "velocity_3d": person.velocity_3d.tolist(),
                "confidence": person.confidence,
                "cameras_visible": person.cameras_visible,
                "last_seen": person.last_seen
            }

            # Add optional fields
            if person.head_pose:
                person_dict["head_pose"] = person.head_pose
            if person.attention_state:
                person_dict["attention_state"] = person.attention_state
            if person.activity:
                person_dict["activity"] = person.activity

            state_dict["persons"].append(person_dict)

        msg = String()
        msg.data = json.dumps(state_dict)
        self.world_state_pub.publish(msg)

    def _publish_markers(self, world_state):
        """Publish visualization markers for persons."""
        marker_array = MarkerArray()

        for person in world_state.persons.values():
            # Create sphere marker for person position
            marker = Marker()
            marker.header = Header()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "world"

            marker.ns = "persons"
            marker.id = person.global_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position
            marker.pose.position.x = float(person.position_3d[0])
            marker.pose.position.y = float(person.position_3d[1])
            marker.pose.position.z = float(person.position_3d[2])
            marker.pose.orientation.w = 1.0

            # Size
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 1.8  # Person height

            # Color (based on confidence)
            marker.color.r = 1.0 - person.confidence
            marker.color.g = person.confidence
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 100000000  # 0.1 second

            marker_array.markers.append(marker)

            # Create arrow marker for velocity
            if np.linalg.norm(person.velocity_3d) > 0.1:
                vel_marker = Marker()
                vel_marker.header = marker.header
                vel_marker.ns = "velocities"
                vel_marker.id = person.global_id + 10000
                vel_marker.type = Marker.ARROW
                vel_marker.action = Marker.ADD

                # Arrow from position in direction of velocity
                vel_marker.points = [
                    Point(
                        x=float(person.position_3d[0]),
                        y=float(person.position_3d[1]),
                        z=float(person.position_3d[2])
                    ),
                    Point(
                        x=float(person.position_3d[0] + person.velocity_3d[0]),
                        y=float(person.position_3d[1] + person.velocity_3d[1]),
                        z=float(person.position_3d[2] + person.velocity_3d[2])
                    )
                ]

                vel_marker.scale.x = 0.1  # Shaft diameter
                vel_marker.scale.y = 0.2  # Head diameter
                vel_marker.scale.z = 0.0

                vel_marker.color.r = 0.0
                vel_marker.color.g = 1.0
                vel_marker.color.b = 1.0
                vel_marker.color.a = 0.8

                vel_marker.lifetime = marker.lifetime

                marker_array.markers.append(vel_marker)

            # Create text marker for ID
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "person_ids"
            text_marker.id = person.global_id + 20000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(person.position_3d[0])
            text_marker.pose.position.y = float(person.position_3d[1])
            text_marker.pose.position.z = float(person.position_3d[2] + 1.0)  # Above head
            text_marker.pose.orientation.w = 1.0

            text_marker.text = f"ID: {person.global_id}"
            text_marker.scale.z = 0.3  # Text height

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.lifetime = marker.lifetime

            marker_array.markers.append(text_marker)

        self.markers_pub.publish(marker_array)

    def _publish_individual_persons(self, world_state):
        """Publish individual person topics."""
        for person in world_state.persons.values():
            person_id = person.global_id

            # Create publishers if needed
            if person_id not in self.person_pose_pubs:
                self.person_pose_pubs[person_id] = self.create_publisher(
                    PoseStamped, f'person/{person_id}/pose', 10
                )
                self.person_velocity_pubs[person_id] = self.create_publisher(
                    Twist, f'person/{person_id}/velocity', 10
                )

            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"

            pose_msg.pose.position.x = float(person.position_3d[0])
            pose_msg.pose.position.y = float(person.position_3d[1])
            pose_msg.pose.position.z = float(person.position_3d[2])

            if person.head_pose:
                quat = self._euler_to_quaternion(
                    person.head_pose.get('roll', 0),
                    person.head_pose.get('pitch', 0),
                    person.head_pose.get('yaw', 0)
                )
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
            else:
                pose_msg.pose.orientation.w = 1.0

            self.person_pose_pubs[person_id].publish(pose_msg)

            # Publish velocity
            vel_msg = Twist()
            vel_msg.linear.x = float(person.velocity_3d[0])
            vel_msg.linear.y = float(person.velocity_3d[1])
            vel_msg.linear.z = float(person.velocity_3d[2])
            # Angular velocity not available
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.0

            self.person_velocity_pubs[person_id].publish(vel_msg)

    def publish_alert(self, alert_type: str, severity: str, message: str):
        """
        Publish alert message.

        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
        """
        alert_dict = {
            "timestamp": time.time(),
            "type": alert_type,
            "severity": severity,
            "message": message
        }

        msg = String()
        msg.data = json.dumps(alert_dict)
        self.alert_pub.publish(msg)

        logger.info(f"Published ROS2 alert: {alert_type} ({severity})")

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles to quaternion.

        Args:
            roll: Roll in degrees
            pitch: Pitch in degrees
            yaw: Yaw in degrees

        Returns:
            Quaternion as [x, y, z, w]
        """
        # Convert to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([x, y, z, w])


class ROS2Bridge:
    """
    ROS2 bridge manager for OMNISENSE.

    Manages ROS2 node lifecycle and integration with OMNISENSE application.
    """

    def __init__(self, config: ROS2Config, app_instance=None):
        """
        Initialize ROS2 bridge.

        Args:
            config: ROS2 configuration
            app_instance: OMNISENSE application instance
        """
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 (rclpy) not installed. Install with: pip install rclpy")

        self.config = config
        self.app_instance = app_instance

        # ROS2 node
        self.node: Optional[OmniSenseROS2Node] = None

        # Spin thread
        self._spin_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info("ROS2 bridge initialized")

    def start(self):
        """Start ROS2 bridge."""
        try:
            # Initialize ROS2
            rclpy.init()

            # Create node
            self.node = OmniSenseROS2Node(self.config, self.app_instance)

            # Start spinning in background thread
            self._stop_event.clear()
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()

            logger.info("ROS2 bridge started")

        except Exception as e:
            logger.error(f"Failed to start ROS2 bridge: {e}")
            raise

    def stop(self):
        """Stop ROS2 bridge."""
        if self.node:
            logger.info("Stopping ROS2 bridge...")

            # Stop spin thread
            self._stop_event.set()
            if self._spin_thread:
                self._spin_thread.join(timeout=5)

            # Destroy node
            self.node.destroy_node()

            # Shutdown ROS2
            rclpy.shutdown()

            logger.info("ROS2 bridge stopped")

    def _spin_loop(self):
        """Background ROS2 spin loop."""
        while not self._stop_event.is_set():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception as e:
                logger.error(f"Error in ROS2 spin loop: {e}")
                time.sleep(0.1)

    def publish_alert(self, alert_type: str, severity: str, message: str):
        """
        Publish alert through ROS2.

        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
        """
        if self.node:
            self.node.publish_alert(alert_type, severity, message)
