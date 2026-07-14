"""Software-only wristband + BLE gateway simulator.

Fills a gap left deliberately open elsewhere in this repo:
``irix.fusion.imu_stream.LiveBLEIMUStream`` and the ``ble_reader``
callables ``irix.live.station_runner``/``irix.live.gym_runner`` expect
are documented stubs, on purpose -- which real BLE GATT client library
and wristband firmware protocol a production deployment uses is a
hardware decision this software scaffold has no way to guess at
correctly (see those modules' docstrings).

That's a different problem from "how do we exercise the live pipeline --
multiple concurrent wristbands, station handoff, packet loss, a radio
dropout and recovery -- before any real hardware exists." This package
answers that second question with pure software: ``SimulatedBLEGateway``
generates the same shapes (``BLEReading``, ``IMUSample`` via the
``IMUStream`` protocol) a real gateway would, with configurable BLE
packet loss and scriptable disconnects, so ``StationSessionRunner``/
``GymSessionRunner`` can be driven end-to-end in CI and demos exactly the
way they'd be driven in production -- see
``irix.demo.run_live_gym_demo`` for the runnable example.

``calibration.py`` is the other piece a real wristband needs before its
IMU readings mean anything: a standard stationary accel/gyro bias
calibration, run once against a batch of samples (real or simulated).
"""
