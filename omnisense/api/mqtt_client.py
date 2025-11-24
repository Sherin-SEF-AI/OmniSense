"""
MQTT Client for OMNISENSE platform.

Provides MQTT publish/subscribe interface for IoT integration.
Publishes real-time updates on person tracking, alerts, and system status.
"""

import json
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import time

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MQTTConfig:
    """MQTT configuration."""
    broker_host: str = "localhost"
    broker_port: int = 1883
    client_id: str = "omnisense"
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: bool = False
    qos: int = 1
    retain: bool = False
    keepalive: int = 60

    # Topic configuration
    base_topic: str = "omnisense"
    world_state_topic: str = "world_state"
    person_topic: str = "persons"
    alert_topic: str = "alerts"
    status_topic: str = "status"


class MQTTClient:
    """
    MQTT client for publishing OMNISENSE data.

    Publishes real-time updates on:
    - World state (all tracked persons)
    - Individual person updates
    - System alerts
    - Status information
    """

    def __init__(self, config: MQTTConfig, app_instance=None):
        """
        Initialize MQTT client.

        Args:
            config: MQTT configuration
            app_instance: OMNISENSE application instance
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt not installed. Install with: pip install paho-mqtt")

        self.config = config
        self.app_instance = app_instance
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self._stop_event = threading.Event()
        self._publish_thread: Optional[threading.Thread] = None

        # Message callbacks
        self._message_callbacks: Dict[str, Callable] = {}

        logger.info("MQTT client initialized")

    def connect(self):
        """Connect to MQTT broker."""
        try:
            # Create MQTT client
            self.client = mqtt.Client(client_id=self.config.client_id)

            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Set credentials if provided
            if self.config.username and self.config.password:
                self.client.username_pw_set(self.config.username, self.config.password)

            # Configure TLS if enabled
            if self.config.use_tls:
                self.client.tls_set()

            # Connect to broker
            logger.info(f"Connecting to MQTT broker at {self.config.broker_host}:{self.config.broker_port}...")
            self.client.connect(
                self.config.broker_host,
                self.config.broker_port,
                self.config.keepalive
            )

            # Start network loop in background
            self.client.loop_start()

            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.connected:
                raise ConnectionError("Failed to connect to MQTT broker within timeout")

            logger.info("Connected to MQTT broker successfully")

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            logger.info("Disconnecting from MQTT broker...")
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from MQTT broker")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection."""
        if rc == 0:
            self.connected = True
            logger.info("MQTT connection established")

            # Subscribe to control topics
            control_topic = f"{self.config.base_topic}/control/#"
            self.client.subscribe(control_topic, qos=self.config.qos)
            logger.info(f"Subscribed to {control_topic}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback for disconnection."""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (code {rc})")
        else:
            logger.info("MQTT client disconnected")

    def _on_message(self, client, userdata, msg):
        """Callback for received messages."""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')

            logger.debug(f"Received MQTT message on {topic}: {payload}")

            # Parse JSON payload
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in MQTT message on {topic}")
                return

            # Route to callback if registered
            for topic_pattern, callback in self._message_callbacks.items():
                if mqtt.topic_matches_sub(topic_pattern, topic):
                    try:
                        callback(topic, data)
                    except Exception as e:
                        logger.error(f"Error in MQTT message callback for {topic}: {e}")

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def publish(self, topic: str, payload: Dict[str, Any], retain: Optional[bool] = None):
        """
        Publish message to MQTT topic.

        Args:
            topic: Topic to publish to (will be prefixed with base_topic)
            payload: Message payload (will be JSON serialized)
            retain: Retain message on broker (uses config default if None)
        """
        if not self.connected:
            logger.warning("Cannot publish: not connected to MQTT broker")
            return

        try:
            # Build full topic
            full_topic = f"{self.config.base_topic}/{topic}"

            # Serialize payload
            message = json.dumps(payload)

            # Publish
            retain_flag = retain if retain is not None else self.config.retain
            result = self.client.publish(
                full_topic,
                message,
                qos=self.config.qos,
                retain=retain_flag
            )

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish to {full_topic}: {result.rc}")
            else:
                logger.debug(f"Published to {full_topic}")

        except Exception as e:
            logger.error(f"Error publishing MQTT message: {e}")

    def subscribe(self, topic_pattern: str, callback: Callable):
        """
        Subscribe to topic pattern with callback.

        Args:
            topic_pattern: Topic pattern (with wildcards)
            callback: Callback function(topic: str, data: dict)
        """
        full_pattern = f"{self.config.base_topic}/{topic_pattern}"
        self._message_callbacks[full_pattern] = callback

        if self.connected:
            self.client.subscribe(full_pattern, qos=self.config.qos)
            logger.info(f"Subscribed to {full_pattern}")

    def start_publishing(self, publish_rate: float = 1.0):
        """
        Start background thread for periodic publishing.

        Args:
            publish_rate: Publishing rate in Hz
        """
        if self._publish_thread and self._publish_thread.is_alive():
            logger.warning("Publishing thread already running")
            return

        self._stop_event.clear()
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            args=(publish_rate,),
            daemon=True
        )
        self._publish_thread.start()
        logger.info(f"Started MQTT publishing at {publish_rate} Hz")

    def stop_publishing(self):
        """Stop background publishing thread."""
        if self._publish_thread:
            self._stop_event.set()
            self._publish_thread.join(timeout=5)
            logger.info("Stopped MQTT publishing")

    def _publish_loop(self, publish_rate: float):
        """Background publishing loop."""
        interval = 1.0 / publish_rate

        while not self._stop_event.is_set():
            try:
                # Publish world state
                self._publish_world_state()

                # Publish system status
                self._publish_status()

                # Sleep
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in MQTT publish loop: {e}")
                time.sleep(interval)

    def _publish_world_state(self):
        """Publish current world state."""
        if not self.app_instance or not self.app_instance.sensor_fusion:
            return

        try:
            world_state = self.app_instance.sensor_fusion.get_world_state()

            # Build payload
            payload = {
                "timestamp": world_state.timestamp,
                "person_count": len(world_state.persons),
                "persons": []
            }

            for person in world_state.persons.values():
                person_data = {
                    "global_id": person.global_id,
                    "position_3d": person.position_3d.tolist(),
                    "velocity_3d": person.velocity_3d.tolist(),
                    "confidence": person.confidence,
                    "cameras_visible": person.cameras_visible,
                    "last_seen": person.last_seen,
                }

                # Add optional fields
                if person.head_pose:
                    person_data["head_pose"] = person.head_pose
                if person.attention_state:
                    person_data["attention_state"] = person.attention_state
                if person.activity:
                    person_data["activity"] = person.activity

                payload["persons"].append(person_data)

            # Publish
            self.publish(self.config.world_state_topic, payload)

        except Exception as e:
            logger.error(f"Error publishing world state: {e}")

    def _publish_status(self):
        """Publish system status."""
        if not self.app_instance:
            return

        try:
            status = self.app_instance.get_status()
            self.publish(self.config.status_topic, status)

        except Exception as e:
            logger.error(f"Error publishing status: {e}")

    def publish_alert(self, alert_type: str, severity: str, message: str, **kwargs):
        """
        Publish alert message.

        Args:
            alert_type: Type of alert
            severity: Severity level (info, warning, error, critical)
            message: Alert message
            **kwargs: Additional alert data
        """
        payload = {
            "timestamp": time.time(),
            "type": alert_type,
            "severity": severity,
            "message": message,
            **kwargs
        }

        self.publish(self.config.alert_topic, payload, retain=True)
        logger.info(f"Published alert: {alert_type} ({severity})")

    def publish_person_update(self, person_id: int, person_data: Dict[str, Any]):
        """
        Publish update for specific person.

        Args:
            person_id: Global person ID
            person_data: Person data dictionary
        """
        topic = f"{self.config.person_topic}/{person_id}"
        self.publish(topic, person_data)
