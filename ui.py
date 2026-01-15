import cv2
import pygame

class Button:
    def __init__(self, x, y, width, height, color=(0, 0, 0), active_color=(0, 255, 0), sounds=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.active_color = active_color
        self.sounds = sounds or []
        self.channels = [sound.play() for sound in self.sounds]
        for channel in self.channels:
            if channel:
                channel.pause()


        self.pinched = {"Left": False, "Right": False}
        self.on = False

    def contains(self, pos):
        """Check if a position is inside this button."""
        if pos is None:
            return False

        x, y = pos
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def update(self, hand, pos):
        """
        Update button state based on finger position.
        Returns True only on the frame the finger ENTERS the button (edge detection).
        """
        inside = self.contains(pos)

        if inside and not self.pinched[hand]:
            self.pinched[hand] = True
            if not self.on:
                self.on = True
                self.play()
            else:
                self.on = False
                self.pause()
            
            return True
        elif not inside:
            self.pinched[hand] = False

        return False

    def draw(self, frame):
        """Draw the button on the frame."""
        color = self.active_color if self.on else self.color
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            3
        )
        return frame

    def play(self):
        for channel in self.channels:
            if channel:
                channel.unpause()
        
    

    def pause(self):
        for channel in self.channels:
            if channel:
                channel.pause()
