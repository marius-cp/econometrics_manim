# Execute in Terminal to render:  manim -pql endobias.py EndoBias 
import numpy as np
from manim import *


class EndoBias(ThreeDScene, VectorScene):
    def construct(self):

        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=0.7)



        def right_angler(A, B, C):
            a = np.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
            b = np.sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)
            c = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

            ratio = (b ** 2 + c ** 2 - a ** 2) / (2 * c ** 2)

            e_sq = (np.tan(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))) * (c * ratio)) ** 2

            return [ratio * B[0], ratio * B[1]], e_sq


        #### Endogeneity Bias ##########################################################################################

        axes = ThreeDAxes()

         # Set axes
        axes1 = Axes(
            x_range=(-0.5, 10, 1),
            y_range=(-0.5, 10, 1),
            x_length=6,
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
            },
            y_axis_config={
                "include_tip": True,
            }
        ).to_edge(LEFT).add_coordinates()
        # Create axes and corresponding labels
        labels = axes1.get_axis_labels(
            x_label=Tex("$x$"), y_label=Tex("$y$")
        )

        axes2 = Axes(
            x_range=(-.5, 10, 1),
            y_range=(-.5, 10, 1),
            x_length=6,
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
            },
            y_axis_config={
                "include_tip": True,
            }
        ).to_edge(RIGHT).add_coordinates()
        # Create axes and corresponding labels
        labels = axes2.get_axis_labels(
            x_label=Tex("$x$"), y_label=Tex("$y$")
        )

        resl = ([3 - 9 * 0.4, 7 - 3 * 0.4])



        # Create the predictor space and show it as a 2D grid in 3D space
        numberplane = NumberPlane(
            y_axis_config={
                "unit_size": axes.get_y_unit_size()
            },
            x_axis_config={
                "unit_size": axes.get_x_unit_size()
            },
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 1,
                "stroke_opacity": 0.6

            }
        )

        # Create axes and corresponding labels
        lab_x = axes.get_x_axis_label(Tex("$x_1$"))
        lab_y = axes.get_y_axis_label(Tex("$x_2$"))

        right_angle_point1, e1 = right_angler(A=[0, 0], B=[10, 5.333], C=[9, 3])
        right_angle_line1 = DashedLine(start=axes1.coords_to_point(9, 3),
                                       end=axes1.coords_to_point(right_angle_point1[0], right_angle_point1[1]),
                                       dash_length=0.1,
                                       color=TEAL)

        right_angle_point2, e2 = right_angler(A=[0, 0], B=[10, 5.333], C=[3, 7])
        right_angle_line2 = DashedLine(start=axes1.coords_to_point(3, 7),
                                       end=axes1.coords_to_point(right_angle_point2[0], right_angle_point2[1]),
                                       dash_length=0.1,
                                       color=TEAL)

        reg_linel = Line(start=axes1.coords_to_point(0, 0), end=axes1.coords_to_point(10, 4), color=RED)
        xb_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9 * 0.35, 3 * 0.35), buff=-1,
                          color=GOLD)
        xb_text = MathTex("\\vec{\\beta}\\vec{x}", color=GOLD).next_to(xb_vector.get_end(), direction=UL)
        xbl = VGroup(xb_vector, xb_text)
        regrl = VGroup(reg_linel, xbl)

        e_vector = Arrow(start=axes2.coords_to_point(9 * 0.35, 3 * 0.35), end=axes2.coords_to_point(3, 7), buff=-1,
                         color=RED)
        e_text = MathTex("\\vec{e}", color=RED).move_to(axes2.coords_to_point(4, 6))
        el = VGroup(e_vector, e_text)


        right_angle_point1, e1 = right_angler(A=[0, 0], B=[10, 4], C=[9, 3])
        right_angle_line1l = DashedLine(start=axes1.coords_to_point(9, 3),
                                       end=axes1.coords_to_point(right_angle_point1[0], right_angle_point1[1]),
                                       dash_length=0.1,
                                       color=TEAL)

        right_angle_point2, e2 = right_angler(A=[0, 0], B=[10, 4], C=[3, 7])
        right_angle_line2l = DashedLine(start=axes1.coords_to_point(3, 7),
                                       end=axes1.coords_to_point(right_angle_point2[0], right_angle_point2[1]),
                                       dash_length=0.1,
                                       color=TEAL)
            
        orthol = VGroup(right_angle_line1l, right_angle_line2l, el)


        x_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9, 3), buff=-1)
        x_text = MathTex("\\vec{x}").next_to(x_vector.get_end(), direction=DOWN)
        xl = VGroup(x_vector, x_text)

        theta_angle = Angle(x_vector, e_vector, radius=.75, other_angle=False)
        theta_text = MathTex("<90^{\\circ}").move_to(axes2.coords_to_point(5, 3))
        thetal = VGroup(theta_angle, theta_text)

        MSE_text = MathTex("\\text{MSE} = ").move_to(axes1.coords_to_point(6, 10))
        MSE_val = DecimalNumber(np.mean([resl[0] ** 2, resl[1] ** 2])).next_to(MSE_text)
        MSEl = VGroup(MSE_text, MSE_val)

        x = VGroup(x_vector, x_text)

        y_Dot = Dot3D(axes.coords_to_point(3, 4, 5), color=BLUE)
        y = Vector(axes.coords_to_point(3, 4, 5), color=BLUE)
        y_name = MathTex("y_0").move_to([2.5, 3.5, 4]).set_color(BLUE).scale(1.25)

        p1_point = Dot(axes1.coords_to_point(9, 3), radius=0.1, color=WHITE)
        p1_text = MathTex("(y_1, x_1)", color=WHITE).next_to(p1_point, direction=DOWN)
        p1 = VGroup(p1_point, p1_text)

        p2_point = Dot(axes1.coords_to_point(3, 7), radius=0.1, color=WHITE)
        p2_text = MathTex("(y_2, x_2)", color=WHITE).next_to(p2_point, direction=UP)
        p2 = VGroup(p2_point, p2_text)



        self.play(GrowFromCenter(VGroup(numberplane, axes, lab_x, lab_y)))
        # Rotate camera
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)

        # Create error vector
        e2 = Line(start=axes.coords_to_point(3, 4, 0), end=axes.coords_to_point(4, 3, 5), stroke_color=GREEN).add_tip()
        e_name2 = MathTex("e_0").move_to([3.5, 4.5, 3]).set_color(GREEN).scale(1.25)

        self.play(Write(e2), run_time=1)
        self.add_fixed_orientation_mobjects(e_name2)
        self.wait(3)

        # Create x1*b1
        b1x1_2 = Dot3D(axes.coords_to_point(3, 0, 0), color=RED, radius=0.1)
        b1x1_name_2 = MathTex("x_{1,0}\\beta_1").next_to(
            [1.75, -1, 0]).set_color(RED).scale(1.25)

        # Create x2*b2
        b2x2_2 = Dot3D(axes.coords_to_point(0, 4, 0), color=RED, radius=0.1)
        b2x2_name_2 = MathTex("x_{2,0}\\beta_2").next_to(
            [-3, 4, 0]).set_color(RED).scale(1.25)

        # Draw the two vectors and afterwards colour them white
        # Create observation vector y
        y2_Dot = Dot3D(axes.coords_to_point(4, 3, 5), color=BLUE, radius=0.1)
        y2 = Vector(axes.coords_to_point(4, 3, 5), color=BLUE)
        y_name2 = MathTex("y_0").move_to([3, 2, 4]).set_color(BLUE).scale(1.25)

        # Draw dashed lines pointing towards the tip of the vector x1b1 + x2b2
        lines = axes.get_lines_to_point(axes.c2p(3, 4, 0))

        self.play(Write(VGroup(y2, y2_Dot, b1x1_name_2, b2x2_name_2, b1x1_2, b2x2_2, lines[0], lines[1])), run_time=1)
        self.add_fixed_orientation_mobjects(y_name2)
        self.wait(5)

        # Create error vector
        e3 = Line(start=axes.coords_to_point(4, 3, 0), end=axes.coords_to_point(4, 3, 5), stroke_color=RED).add_tip()
        e_name3 = MathTex("\\hat{e}_0").move_to([4.5, 3.5, 2]).set_color(RED).scale(1.25)

        # Indicate the right Angle
        curve2 = ParametricFunction(
            lambda u: np.array([
                (1.5 * np.sin(u) * np.cos(0.92729343) + 4) * axes.get_x_unit_size(),
                (1.5 * np.sin(u) * np.sin(0.92729343) + 3) * axes.get_y_unit_size(),
                (1.5 * np.cos(u))
            ]), color=WHITE, t_range=[3 * TAU / 4, TAU]
        ).set_shade_in_3d(True)
        # degree = MathTex("90^\\circ").next_to([1.5, 3.75, 0.5], UL, buff=0).set_color(WHITE).scale(1)
        degree2 = Dot3D(axes.coords_to_point((-1 * np.sin(3 * TAU / 8) * np.cos(0.92729343) + 4),
                                            (-1 * np.sin(3 * TAU / 8) * np.sin(0.92729343) + 3),
                                            (-1 * np.cos(3 * TAU / 8))), radius=0.1)

        self.play(Write(VGroup(e3, curve2, degree2)))
        self.add_fixed_orientation_mobjects(e_name3)
        self.begin_ambient_camera_rotation(rate=0.5)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(y2, curve2, degree2, y_name2))

        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.wait()

        # Draw dashed lines pointing towards the tip of the vector x1b1 + x2b2
        lines = axes.get_lines_to_point(axes.c2p(4, 3, 0))
        self.play(Write(lines[0]), Write(lines[1]), run_time=1)
        self.wait()

        # Create x1*b1
        b1x1_3 = Dot3D(axes.coords_to_point(4, 0, 0), color=YELLOW)
        b1x1_name_3 = MathTex("x_{1,0}\\hat{\\beta}_1").next_to(
            [1.75, 1, 0]).set_color(YELLOW).scale(1.25)

        # Create x2*b2
        b2x2_3 = Dot3D(axes.coords_to_point(0, 3, 0), color=YELLOW, )
        b2x2_name_3 = MathTex("x_{2,0}\\hat{\\beta}_2").next_to(
            [-3, 3, 0]).set_color(YELLOW).scale(1.25)

        # Draw the two vectors and afterwards colour them white
        self.play(Write(b1x1_3), Write(b2x2_3),
                  run_time=1)
        self.play(Write(VGroup(b1x1_name_3, b2x2_name_3)))
        self.wait(5)
