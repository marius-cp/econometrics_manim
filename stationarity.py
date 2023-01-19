# Execute in Terminal to render: manim -pql stationarity.py Stationarity
import numpy as np
from manim import *
import numpy as np

class Stationarity(MovingCameraScene):
    def construct(self):

        # Set axes
        axes = Axes(
            x_range=(-5, 100, 10),
            y_range=(-5, 5, 0.5),
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
            },
            y_axis_config={
                "include_tip": False,
            }
        )
        # Create axes and corresponding labels
        labels = axes.get_axis_labels(
            x_label=Tex("$t$"), y_label=Tex("$y_t$")
        )

        # Write intro and transform it cool
        intro = Text("Stationarity of Random Processes")
        self.add(intro)
        self.wait()
        self.play(FadeOut(intro))
        self.wait()

        # Introduction texts texts
        intro_text3 = MathTex("y_t=\\sum_{i = 1}^p", "\\phi_i", "y_{t-i} +", "\\epsilon_t.").shift(UP)
        self.play(Write(intro_text3))
        self.wait(2)

        # Create a highlighting box
        box1 = SurroundingRectangle(intro_text3[3])

        self.play(Create(box1))

        # Create a highlighting box
        box2 = SurroundingRectangle(intro_text3[1])

        self.play(ReplacementTransform(box1, box2))

        labelings = VGroup(axes, labels)

        self.wait(2)


        self.play(ReplacementTransform(VGroup(intro_text3, box2), labelings))


        # Show different kind of stationary processes

        # Set seed
        np.random.seed(19052022)

        # Create error terms
        e = np.random.standard_normal(100)

        # Create observations as a graph
        ## Stationary
        y05 = [0]
        for t in range(1, 99):
            y05.append(0.5 * y05[t - 1] + e[t])
        y05 = list(y05)

        # Extract the points for the stationary process
        statio = ParametricFunction(
            lambda t: np.array([(t - 47.5) * axes.get_x_unit_size(),
                                (y05[t] if type(t) == int else (y05[np.ceil(t).astype(int)] - y05[
                                    np.floor(t).astype(int)]) * (t - np.floor(t)) + y05[
                                                                   np.floor(t).astype(int)]) * axes.get_y_unit_size(),
                                0]), color=GREEN, t_range=[0, 98]
        )

        ## Unit Root
        y1 = [0]
        for t in range(1, 99):
            y1.append(1 * y1[t - 1] + e[t])
        y1 = list(y1)

        # Extract the points for the stationary process
        unit = ParametricFunction(
            lambda t: np.array([(t - 47.5) * axes.get_x_unit_size(),
                                (y1[t] if type(t) == int else (y1[np.ceil(t).astype(int)] - y1[
                                    np.floor(t).astype(int)]) * (t - np.floor(t)) + y1[
                                                                   np.floor(t).astype(int)]) * axes.get_y_unit_size(),
                                0]), color=YELLOW, t_range=[0, 98]
        )

        ## Exponential
        y11 = [0]
        for t in range(1, 99):
            y11.append(1.1 * y11[t - 1] + e[t])
        y11 = list(y11)

        # Extract the points for the stationary process
        expo = ParametricFunction(
            lambda t: np.array([(t - 47.5) * axes.get_x_unit_size(),
                                (y11[t] if type(t) == int else (y11[np.ceil(t).astype(int)] - y11[
                                    np.floor(t).astype(int)]) * (t - np.floor(t)) + y11[
                                                                  np.floor(t).astype(int)]) * axes.get_y_unit_size(),
                                0]), color=RED, t_range=[0, 98]
        )

        # Draw the process
        # Introduce the imaginary plane
        # Move the camera
        self.play(self.camera.frame.animate.set(width=axes.width * 1.5).move_to([1.75 * UR + 1 * RIGHT]))

        self.wait()

        # Create the center point of the unit circle
        Im_center = [80 * axes.get_x_unit_size(), 6 * axes.get_y_unit_size(), 0]

        # Create the unit circle plain
        ## Circle
        circ = Circle(radius=1.5, color=WHITE).move_to(Im_center)
        ## Real line
        Re = Line([0, 0, 0], [5, 0, 0]).move_to(Im_center)
        Re_lab = Tex("$\\mathfrak{Re}$").next_to([(80) * axes.get_x_unit_size() + 1.5,
                                                 (6) * axes.get_y_unit_size() - 0.5,
                                                 0])
        ## Imaginary Line
        Im = Line([0, 0, 0], [0, 5, 0]).move_to(Im_center)
        Im_lab = Tex("$\\mathfrak{Im}$").next_to([(80) * axes.get_x_unit_size() + 0.1,
                                                  (6) * axes.get_y_unit_size() + 2,
                                                  0])
        ## Numberplane
        numberplane = NumberPlane(
            x_range=[-2.5, 2.5, 0.5],
            y_range=[-2.5, 2.5, 0.5],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 1,
                "stroke_opacity": 0.6

            }
        ).move_to(Im_center)
        ## Draw
        self.play(GrowFromPoint(VGroup(Re, Re_lab, Im, Im_lab, numberplane), Im_center))


        self.play(GrowFromPoint(circ, Im_center))


        # Stationary case
        self.play(self.camera.frame.animate.set(width=axes.width * 1.5).move_to(Im_center))


        # Create the root indicators
        root1 = Dot(color=GREEN).shift([80 * axes.get_x_unit_size() + 1, 6 * axes.get_y_unit_size() + 0.333, 0])
        root2 = Dot(color=GREEN).shift([80 * axes.get_x_unit_size() + 1, 6 * axes.get_y_unit_size() - 0.333, 0])
        root3 = Dot(color=GREEN).shift([80 * axes.get_x_unit_size() - 0.7, 6 * axes.get_y_unit_size() + 1, 0])
        root4 = Dot(color=GREEN).shift([80 * axes.get_x_unit_size() - 0.7, 6 * axes.get_y_unit_size() - 1, 0])

        self.play(GrowFromPoint(VGroup(root1, root2, root3, root4), Im_center))
        self.wait()

        self.play(self.camera.frame.animate.set(width=axes.width * 1.5).move_to([1.75 * UR + 1 * RIGHT]))



        self.play(Write(statio))
        self.wait(2)

    

        # Unit root case
        #unit_text1 = Tex(
        #    "Now, let's move the solutions\\\\ of the polynomial.").shift(
        #    [25 * axes.get_x_unit_size(), 4.5 * axes.get_y_unit_size(), 0])
        #self.add(unit_text1)
        #self.wait(1)
        self.play(statio.animate.set_color(WHITE), root3.animate.shift([0, (1.5**2 - 0.7**2)**0.5 - 1, 0]).set_color(YELLOW),
                  root4.animate.shift([0,  - (1.5**2 - 0.7**2)**0.5 + 1, 0]).set_color(YELLOW), root1.animate.set_color(YELLOW),
                  root2.animate.set_color(YELLOW))
        self.wait(2)

        #self.remove(unit_text1)

        self.play(Write(unit), self.camera.frame.animate.set(width=axes.width * 2))

        self.play(unit.animate.set_color(WHITE),
                  root3.animate.shift([0, (1.5 ** 2 - 0.7 ** 2) ** 0.5 - 0.99, 0]).set_color(RED),
                  root4.animate.shift([0, - (1.5 ** 2 - 0.7 ** 2) ** 0.5 + 0.99, 0]).set_color(RED),
                  root1.animate.set_color(RED),
                  root2.animate.set_color(RED))
        self.wait(1)

        self.play(Write(expo))

        self.play(self.camera.frame.animate.set(width=axes.width * 10))
        self.wait()
        self.play(self.camera.frame.animate.set(width=axes.width * 1.5))
        self.wait(5)








