"""
Database schema for OMNISENSE platform using PostgreSQL + TimescaleDB.

Defines tables for storing time-series perception data, tracks, events, and analytics.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    DateTime, JSON, LargeBinary, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import numpy as np

from omnisense.utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class CameraFrame(Base):
    """Camera frame metadata."""
    __tablename__ = 'camera_frames'

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    frame_number = Column(Integer)
    resolution_width = Column(Integer)
    resolution_height = Column(Integer)
    file_path = Column(String)  # Path to saved frame image


class PersonTrack(Base):
    """Person tracking data."""
    __tablename__ = 'person_tracks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, nullable=False, index=True)
    camera_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    # 2D bounding box
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

    # 3D position in global frame
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)

    # Velocity
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    velocity_z = Column(Float)

    confidence = Column(Float)

    # Appearance features (binary blob)
    features = Column(LargeBinary)


class FaceState(Base):
    """Face analysis state."""
    __tablename__ = 'face_states'

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, nullable=False, index=True)
    camera_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Head pose
    roll = Column(Float)
    pitch = Column(Float)
    yaw = Column(Float)

    # Gaze
    gaze_x = Column(Float)
    gaze_y = Column(Float)
    gaze_z = Column(Float)

    # Eye state
    eye_aspect_ratio = Column(Float)


class BodyPose(Base):
    """Body pose data."""
    __tablename__ = 'body_poses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, nullable=False, index=True)
    camera_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    posture_type = Column(String)
    spine_angle = Column(Float)
    shoulder_alignment = Column(Float)

    # Keypoints stored as JSON
    keypoints = Column(JSON)


class DrowsinessEvent(Base):
    """Drowsiness detection events."""
    __tablename__ = 'drowsiness_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    perclos = Column(Float)
    blink_rate = Column(Float)
    alert_level = Column(String)  # normal, warning, critical
    is_drowsy = Column(Boolean)


class Activity(Base):
    """Detected activities."""
    __tablename__ = 'activities'

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    activity_type = Column(String, index=True)
    confidence = Column(Float)
    duration = Column(Float)  # seconds

    # Additional context as JSON
    context = Column(JSON)


class Interaction(Base):
    """Person-to-person or person-to-object interactions."""
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    interaction_type = Column(String)
    person_ids = Column(JSON)  # Array of involved person IDs
    confidence = Column(Float)

    # Spatial context
    location_x = Column(Float)
    location_y = Column(Float)
    location_z = Column(Float)


class Alert(Base):
    """System alerts and notifications."""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    alert_type = Column(String, index=True)
    severity = Column(String)  # info, warning, critical
    message = Column(String)

    # Related entities
    person_id = Column(Integer)
    camera_id = Column(Integer)

    # Additional data
    data = Column(JSON)

    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)


class OccupancyHistory(Base):
    """Spatial occupancy over time."""
    __tablename__ = 'occupancy_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    zone_id = Column(String, index=True)
    person_count = Column(Integer)
    person_ids = Column(JSON)


class SystemMetrics(Base):
    """System performance metrics."""
    __tablename__ = 'system_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    metric_name = Column(String, index=True)
    metric_value = Column(Float)

    # Context
    camera_id = Column(Integer)
    module_name = Column(String)


# Create indices for time-series queries
Index('idx_person_tracks_time', PersonTrack.timestamp)
Index('idx_face_states_time', FaceState.timestamp)
Index('idx_activities_time', Activity.timestamp)
Index('idx_alerts_time', Alert.timestamp)


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, connection_string: str):
        """
        Initialize database manager.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.engine = create_engine(connection_string, pool_size=10, max_overflow=20)
        self.Session = sessionmaker(bind=self.engine)

        logger.info(f"DatabaseManager initialized")

    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def create_hypertables(self):
        """
        Convert tables to TimescaleDB hypertables for time-series optimization.

        Requires TimescaleDB extension to be installed.
        """
        session = self.Session()

        try:
            # Convert time-series tables to hypertables
            tables_to_convert = [
                'person_tracks',
                'face_states',
                'body_poses',
                'drowsiness_events',
                'activities',
                'interactions',
                'alerts',
                'occupancy_history',
                'system_metrics'
            ]

            for table in tables_to_convert:
                try:
                    session.execute(
                        f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE)"
                    )
                    logger.info(f"Created hypertable for {table}")
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {table}: {e}")

            session.commit()
            logger.info("Hypertables created successfully")

        except Exception as e:
            logger.error(f"Error creating hypertables: {e}")
            session.rollback()
        finally:
            session.close()

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    def insert_person_track(self, session, track_data: dict):
        """Insert person track data."""
        track = PersonTrack(**track_data)
        session.add(track)

    def insert_alert(self, session, alert_data: dict):
        """Insert alert."""
        alert = Alert(**alert_data)
        session.add(alert)

    def query_recent_alerts(self, session, limit: int = 100):
        """Query recent alerts."""
        return session.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()

    def cleanup_old_data(self, session, days: int = 30):
        """Delete data older than specified days."""
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)

        # Delete old records from each table
        session.query(CameraFrame).filter(CameraFrame.timestamp < cutoff_date).delete()
        session.query(PersonTrack).filter(PersonTrack.timestamp < cutoff_date).delete()
        session.query(FaceState).filter(FaceState.timestamp < cutoff_date).delete()
        session.query(BodyPose).filter(BodyPose.timestamp < cutoff_date).delete()
        session.query(Activity).filter(Activity.timestamp < cutoff_date).delete()

        session.commit()
        logger.info(f"Cleaned up data older than {days} days")
