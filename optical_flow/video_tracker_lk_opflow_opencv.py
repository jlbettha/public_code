## file may not run in WSL due to OpenCV camera access limitations
from dataclasses import dataclass

import cv2
import numpy as np

# Constants
ESC_KEY = 27
WINDOW_NAME = "Frame"


@dataclass
class TrackerState:
    """Encapsulates the state of the point tracker"""

    point: tuple[int, int] | None = None
    point_selected: bool = False
    old_points: np.ndarray | None = None

    def reset(self) -> None:
        """Reset tracker state"""
        self.point = None
        self.point_selected = False
        self.old_points = None

    def set_point(self, x: int, y: int) -> None:
        """Set the tracking point"""
        self.point = (x, y)
        self.point_selected = True
        self.old_points = np.array([[x, y]], dtype=np.float32)


class VideoTracker:
    """Lucas-Kanade optical flow point tracker"""

    def __init__(self):
        self.state = TrackerState()
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 4,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

    def _mouse_callback(self, event: int, x: int, y: int) -> None:
        """Mouse callback function for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.set_point(x, y)
            print(f"Selected point: ({x}, {y})")

    def run(self) -> None:
        """Main tracking loop"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Create old frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            cap.release()
            return

        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print("Click a point in video to track its movement. Press <ESC> to exit.")

        # Setup window and mouse callback
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        # Colors for tracks
        rng = np.random.default_rng()
        colors = rng.integers(0, 255, (100, 3))
        tracks = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                # If we have a point selected, track it
                if self.state.point_selected and self.state.old_points is not None:
                    # Calculate optical flow
                    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        old_gray, frame_gray, self.state.old_points, None, **self.lk_params
                    )

                    # Select good points
                    if new_points is not None and status is not None:
                        good_new = new_points[status == 1]
                        good_old = self.state.old_points[status == 1]

                        # Draw tracks
                        for i, (new, old) in enumerate(zip(good_new, good_old, strict=True)):
                            a, b = new.ravel().astype(int)
                            c, d = old.ravel().astype(int)

                            # Add to tracks
                            tracks.append((c, d, a, b))

                            # Draw track line
                            cv2.line(img, (a, b), (c, d), colors[i % len(colors)].tolist(), 2)
                            cv2.circle(img, (a, b), 5, colors[i % len(colors)].tolist(), -1)

                        # Update the previous frame and points
                        self.state.old_points = good_new.reshape(-1, 1, 2)

                # Draw all track history
                for track in tracks[-50:]:  # Keep last 50 track points
                    cv2.line(img, (track[0], track[1]), (track[2], track[3]), (0, 255, 0), 1)

                cv2.imshow(WINDOW_NAME, img)

                # Update for next iteration
                old_gray = frame_gray.copy()

                # Check for exit
                key = cv2.waitKey(30) & 0xFF
                if key == ESC_KEY:
                    break
                if key == ord("r"):  # Reset tracking
                    self.state.reset()
                    tracks.clear()
                    print("Tracking reset. Click a new point to track.")

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    """Entry point"""
    tracker = VideoTracker()
    tracker.run()


if __name__ == "__main__":
    main()
