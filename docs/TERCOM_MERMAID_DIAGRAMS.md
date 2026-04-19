# TERCOM Navigation Mermaid Diagrams

This document contains a comprehensive collection of Mermaid diagrams depicting the architecture, states, algorithms, and data flows of the `tercom_nav` package.

## 1. System Architecture Data Flow
Shows the high-level data flow from sensors and DEMs to the ESKF and TERCOM matching nodes.
```mermaid
graph TD
    subgraph SENSORS [Sensor Inputs]
        S1[Barometer] -->|/target/mavros/altitude| N3[TERCOM Node]
        S2[Laser Rangefinder] -->|/distance_sensor| N3
        S3[IMU] -->|/target/mavros/imu/data| N1[ESKF Node]
        S3 -->|/target/mavros/imu/data| N3
        S4[GPS - Init Only] -->|/target/mavros/global_position/global| N1
        S5[Local Velocity] -->|/target/mavros/local_position/velocity_local| N1
        S6[Odometry] -->|/target/mavros/local_position/odom| N3
    end

    subgraph TERRAIN_DATABASE [Terrain Database]
        D1[(GeoTIFF DEM)] --> N4[DEM Manager]
        N4 --> N3
        D1 --> N2[DEM Server Node]
    end

    subgraph CORE_ESTIMATION [Core Estimation]
        N1[ESKF Node] -->|/tercom/eskf_node/pose \n covariance feedback| N3
        N3 -.->|/tercom/tercom_node/position_fix \n quality metrics| N1
    end
    
    subgraph DIAGNOSTICS [Diagnostics]
        N1 --> |/tercom/eskf_node/odom| NX[Diagnostics Node]
        NX --> |log csv & visual| RViz[(RViz2)]
    end
```

## 2. ESKF Node State Machine
The top-level state machine handling the initialization and health of the Error-State Kalman Filter.
```mermaid
stateDiagram-v2
    [*] --> WAITING_GPS
    WAITING_GPS --> INITIALIZING : Acquired GPS Fix
    INITIALIZING --> RUNNING : Collected N GPS Samples
    INITIALIZING --> WAITING_GPS : Timeout before N Samples
    
    RUNNING --> DIVERGED : Covariance Limit Exceeded
    RUNNING --> DIVERGED : Constant High NIS
    
    DIVERGED --> RESETTING : Health Monitor Protocol Action
    RESETTING --> RUNNING : Soft Reset (Reset matrices, keep state)
    RESETTING --> WAITING_GPS : Hard Reset (Scrap state, reboot GPS)
```

## 3. ESKF Filter Lifecycle (Predict & Update)
Detailed algorithmic loop of the Error-State Kalman Filter, outlining nominal state kinematics and error state covariance updates.
```mermaid
graph TD
    A[Start Loop] --> B{Data Received?}
    B -- IMU Msg --> C[Predict Step]
    C --> C1[Nominal State Integration]
    C1 --> C2[Error Covariance Prop P = F*P*F'+Q]
    C2 --> B
    
    B -- TERCOM Fix / Baro / Vel --> D[Update Step]
    D --> D1[Calc Innovation y = z - H*x]
    D1 --> D2[Check NIS & Health bounds]
    D2 --> D3[Calculate Kalman Gain K = P*H'*inv(H*P*H'+R)]
    D3 --> D4[Error State Correction dx = K*y]
    D4 --> D5[Inject dx into Nominal State p, v, q, b]
    D5 --> D6[Update Covariance P = (I-KH)P(I-KH)' + KRK']
    D6 --> D7[Reset Error State dx = 0]
    D7 --> B
```

## 4. TERCOM Sequence Diagram
Interaction between the different components handling profile collection and matching.
```mermaid
sequenceDiagram
    participant S as Sensors (Baro+Laser+IMU)
    participant C as ProfileCollector
    participant M as Vectorized Matcher
    participant E as ESKF Node

    S->>C: h_baro, h_agl, pos_enu, time
    Note over C: Wait until motion creates spacing > min_spacing
    C-->>C: Compute h_terrain & relative dx/dy
    C-->>C: Store sample in array memory
    opt When profile buffer is full (N samples collected)
        C->>M: terrain_h array, dx array, dy array, predicted_utm
        M-->>M: 1. Search Grid Gen (M bounds)
        M-->>M: 2. Matrix Broadcasting (M x N index mappings)
        M-->>M: 3. Mask NoData & Compute MAD scores
        M-->>M: 4. Find min(MAD) and calc Discrimination
        M->>C: Slice sliding window (keep newest N/2 samples)
        M->>E: Publish 2D Position Fix (if health valid)
        M->>E: Publish Adaptive Covariance Quality
    end
```

## 5. Coordinate Map Transformation
Shows the coordinate reference systems and how spatial data is converted in `core/coordinate_utils.py`.
```mermaid
graph LR
    WGS84((WGS84<br>Lat/Lon/Alt)) -- latlon_to_utm --> UTM((UTM<br>Easting/Northing))
    UTM -- utm_to_local_enu --> ENU((Local ENU<br>X/Y/Z))
    UTM -- utm_to_pixel --> PIX((DEM Pixel<br>Col/Row))
    ENU -- local_enu_to_utm --> UTM
```

## 6. Core Modules Class Architecture
Class structure of the Python backend.
```mermaid
classDiagram
    class EskfNode {
        -eskf: ESKF
        -health_monitor: HealthMonitor
        -timer: Timer
    }
    
    class TercomNode {
        -dem_manager: DemManager
        -matcher: TercomMatcher
        -terrain_quality: TerrainQuality
        -adaptive_sampler: AdaptiveSampler
    }
    
    class DemServerNode {
        -dem_manager: DemManager
        +serve_metadata()
    }
    
    class DemManager {
        +get_elevation(x, y)
        +get_elevation_batch(x_arr, y_arr)
        -_load_dem()
    }
    
    class TercomMatcher {
        -profile_collector: ProfileCollector
        +match_profile(...)
    }

    class ProfileCollector {
        +samples: List
        +try_add_sample()
    }

    class ESKF {
        +P: Matrix
        +predict(imu)
        +update_tercom(fix)
        +update_baro(alt)
    }

    class HealthMonitor {
        +check_nis()
        +check_divergence()
    }
    
    EskfNode *-- ESKF
    EskfNode *-- HealthMonitor
    TercomNode *-- DemManager
    TercomNode *-- TercomMatcher
    TercomNode *-- TerrainQuality
    TercomNode *-- AdaptiveSampler
    DemServerNode *-- DemManager
    TercomMatcher *-- ProfileCollector
```
