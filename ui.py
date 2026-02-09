import cv2
import math
import numpy as np

class Button:
    def __init__(self, x, y, width, height, color=(0, 0, 0), active_color=(0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.active_color = active_color
        self.pinched = {"Left": False, "Right": False}
        self.on = False

    def contains(self, pos):
        if pos is None:
            return False
        x, y = pos
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def update(self, hand, pos):
        inside = self.contains(pos)

        if inside and not self.pinched[hand]:
            self.pinched[hand] = True
            if not self.on:
                self.on = True
                self.activate()
            else:
                self.on = False
                self.deactivate()
            return True
        elif not inside:
            self.pinched[hand] = False

        return False

    def draw(self, frame):
        color = self.active_color if self.on else self.color
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            3
        )
        return frame

    def activate(self):
        pass

    def deactivate(self):
        pass


class PlayButton(Button):
    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.selector = selector
        self.side = side

    def activate(self):
        self.selector.play(self.side)

    def deactivate(self):
        self.selector.pause(self.side)


class StemButton(Button):
    def __init__(self, x, y, width, height, selector, side, stem_index, label="", **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.selector = selector
        self.side = side
        self.stem_index = stem_index
        self.label = label
        self.on = True  # stems start unmuted

    def activate(self):
        self.selector.unmute(self.side, self.stem_index)

    def deactivate(self):
        self.selector.mute(self.side, self.stem_index)

    def draw_label(self, frame):
        cv2.putText(frame, self.label,
                    (self.x + 4, self.y + self.height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame


class Deck:
    def __init__(self, cx, cy, radius, selector, side, label="", color=(255, 255, 255)):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.selector = selector
        self.side = side
        self.label = label
        self.color = color
        self.prev_angle = {"Left": None, "Right": None}
        self.angle = 0

    def contains(self, pos):
        if pos is None:
            return False
        x, y = pos
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def update(self, hand, pos):
        inside = self.contains(pos)

        if inside and not self.prev_angle[hand]:
            self.prev_angle[hand] = self.calc_angle(pos)
        elif inside:
            cur_angle = self.calc_angle(pos)
            delta = cur_angle - self.prev_angle[hand]

            # correct for wraparound
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

            self.prev_angle[hand] = cur_angle
            self.angle = (self.angle + delta) % (2 * math.pi)

            seek_value = 1.5 * delta

            self.selector.seek(self.side, seek_value)


        elif not inside:
            self.prev_angle[hand] = None

    def calc_angle(self, pos):
        x, y = pos
        dx = x - self.cx
        dy = y - self.cy
        return math.atan2(dy, dx)

    def draw(self, frame):
        stamp = np.zeros((2 * self.radius, 2 * self.radius, 3), dtype=np.uint8)
        cv2.circle(stamp, (self.radius, self.radius), self.radius, self.color, 2)
        if self.label:
            cv2.putText(stamp, self.label,
                        (self.radius - 10, self.radius + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
            
        degrees = math.degrees(self.angle)
        M = cv2.getRotationMatrix2D((self.radius, self.radius), -degrees, 1.0)

        rotated = cv2.warpAffine(stamp, M, (2 * self.radius, 2 * self.radius))

        x1 = self.cx - self.radius                                                                                                                                                                                
        y1 = self.cy - self.radius                                                                                                                                                                                
        x2 = self.cx + self.radius                                                                                                                                                                                
        y2 = self.cy + self.radius                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                            
        mask = rotated > 30

        frame[y1:y2, x1:x2][mask] = rotated[mask]
        
        return frame

