# Execute in Terminal to render:  manim -pql orthogonality.py Orthogonality 
import numpy as np
from manim import *


class Orthogonality(ThreeDScene, VectorScene):
    def construct(self):

        #### Orthogonality by construction #############################################################################

        # Set default camara orientation
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=0.7)
        axes = ThreeDAxes()

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
        coords = VGroup(axes, lab_x, lab_y)

        intro = Text("Orthogonality of Residuals-Graphical Visualisation")
        self.add(intro)
        self.wait()
        self.play(ReplacementTransform(intro, coords))
        self.wait()

        # Grow predictor space
        self.play(GrowFromCenter(numberplane))
        self.wait(5)

        # Create x1*b1
        b1x1 = Dot3D(axes.coords_to_point(3, 0, 0), color=RED, radius=0.1)
        b1x1_name = MathTex("\\hat{\\beta}_1\cdot x_{1,0}").next_to([1.75, -1, 0]).set_color(RED).scale(1.25)

        # Create x2*b2
        b2x2 = Dot3D(axes.coords_to_point(0, 4, 0), color=RED, radius=0.1)
        b2x2_name = MathTex("\\hat{\\beta}_2\cdot x_{2,0}").next_to([-3, 4, 0]).set_color(RED).scale(1.25)

        self.play(GrowFromPoint(b1x1, point=axes.coords_to_point(3, 0, 0)), Write(b1x1_name), run_time=1)
        self.play(GrowFromPoint(b2x2, point=axes.coords_to_point(0, 4, 0)), Write(b2x2_name), run_time=1)

        #self.play(b1x1.animate.set_color(WHITE), b2x2.animate.set_color(WHITE),
        #          b1x1_name.animate.set_color(WHITE), b2x2_name.animate.set_color(WHITE))

        # Draw dashed lines pointing towards the tip of the vector x1b1 + x2b2
        bx_lines = axes.get_lines_to_point(axes.c2p(3, 4, 0))
        self.play(Write(bx_lines[0]), Write(bx_lines[1]), run_time=1)
        self.wait()

        # Create the y_hat vector
        bx_Dot = Dot3D(axes.coords_to_point(3, 4, 0), color=YELLOW)
        bx_Vec = Vector(axes.coords_to_point(3, 4, 0), color=YELLOW)
        bx_name2 = MathTex("\\hat{y}").move_to([3.5, 4.5, -0.5], UR).set_color(YELLOW).scale(1.25)
        self.play(Write(bx_Dot), run_time=1)
        self.add_fixed_orientation_mobjects(bx_name2)
        self.wait(3)
        self.play(Write(bx_Vec))
        self.wait(2)

        # Rotate camera
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        ## Reset default camera orientation
        # Set default camara orientation
        self.wait(2)

        # Create observation vector y
        y_Dot = Dot3D(axes.coords_to_point(3, 4, 5), color=BLUE)
        y = Vector(axes.coords_to_point(3, 4, 5), color=BLUE)
        y_name = MathTex("y_0").move_to([2.5, 3.5, 4]).set_color(BLUE).scale(1.25)

        # Create error vector
        e = Line(start=axes.coords_to_point(3, 4, 0), end=axes.coords_to_point(3, 4, 5), stroke_color=GREEN).add_tip()
        e_name = MathTex("\\hat{e}_0").move_to(axes.coords_to_point(3.5, 4.5, 2)).set_color(GREEN).scale(1.25)

        # Draw both into the graph
        self.play(Write(y_Dot), run_time=1)
        self.add_fixed_orientation_mobjects(y_name)
        self.wait(3)
        self.play(Write(y))
        self.wait(5)
        self.play(Write(e), run_time=1)
        self.wait(2)
        self.add_fixed_orientation_mobjects(e_name)
        self.wait()
        self.begin_ambient_camera_rotation(rate=0.25)
        # Indicate the right Angle
        curve1 = ParametricFunction(
            lambda u: np.array([
                (1.5 * np.sin(u) * np.cos(0.92729343) + 3) * axes.get_x_unit_size(),
                (1.5 * np.sin(u) * np.sin(0.92729343) + 4) * axes.get_y_unit_size(),
                (1.5 * np.cos(u))
            ]), color=WHITE, t_range=[3 * TAU / 4, TAU]
        ).set_shade_in_3d(True)
        #degree = MathTex("90^\\circ").next_to([1.5, 3.75, 0.5], UL, buff=0).set_color(WHITE).scale(1)
        degree = Dot3D(axes.coords_to_point((-1 * np.sin(3 * TAU / 8) * np.cos(0.92729343) + 3),
                                            (-1 * np.sin(3 * TAU / 8) * np.sin(0.92729343) + 4),
                                            (-1 * np.cos(3 * TAU / 8))), radius=0.1)

        self.play(Write(VGroup(curve1, degree)))
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait(5)

        ## Again rotate camera to view the right angle
        #self.move_camera(phi=90 * DEGREES, theta=0 * DEGREES)
        #self.wait(5)

        #line1 = Line(start=ORIGIN, end=bx_Vec.get_end(), color=GREEN, stroke_width=10)
        #line2 = Line(start=bx_Vec.get_end(), end=e.get_end(), color=GREEN, stroke_width=10)
        #line3 = Line(start=e.get_end(), end=ORIGIN, color=GREEN, stroke_width=10)

        ## Add some last descriptions
        #self.play(FadeOut(VGroup(b1x1, b2x2, y, e, bx_Vec)))
        #lab_z = MathTex("y").next_to([-0.5, -0.5, 3])
        #x_space = Tex("$x_1$-$x_2$-Space").next_to([0, -3, -0.5])
        #self.play(Write(line1), Write(line2), Write(line3))
        #self.add_fixed_orientation_mobjects(lab_z, x_space)
        #self.wait()



        self.remove(#line1, line2, line3,
                    b1x1, b2x2, bx_Dot, bx_Vec, bx_name2, y, y_Dot, y_name, e, e_name,
                    #x_space,
                    curve1,
                    degree, b1x1_name, b2x2_name, bx_lines[0], bx_lines[1])

        #### Right-Angle Video-Pics ####################################################################################

        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)

        ## just right

        # Helper Functions
        ## Find the right angle for points to the regression line
        def right_angler(A, B, C):
            a = np.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
            b = np.sqrt((C[0] - A[0]) ** 2 + (C[1] - A[1]) ** 2)
            c = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

            ratio = (b ** 2 + c ** 2 - a ** 2) / (2 * c ** 2)

            e_sq = (np.tan(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))) * (c * ratio)) ** 2

            return [ratio * B[0], ratio * B[1]], e_sq

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

        # Create the "observations"
        p1_point = Dot(axes1.coords_to_point(9, 3), radius=0.1, color=WHITE)
        p1_text = MathTex("(y_1, x_1)", color=WHITE).next_to(p1_point, direction=DOWN)
        p1 = VGroup(p1_point, p1_text)

        p2_point = Dot(axes1.coords_to_point(3, 7), radius=0.1, color=WHITE)
        p2_text = MathTex("(y_2, x_2)", color=WHITE).next_to(p2_point, direction=UP)
        p2 = VGroup(p2_point, p2_text)

        x_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9, 3), buff=-1)
        x_text = MathTex("\\vec{x}").next_to(x_vector.get_end(), direction=DOWN)
        x = VGroup(x_vector, x_text)

        y_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(3, 7), buff=-1)
        y_text = MathTex("\\vec{y}").next_to(y_vector.get_end(), direction=LEFT)
        y = VGroup(y_vector, y_text)

        xb_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9 * 0.5333, 3 * 0.5333), buff=-1,
                          color=GOLD)
        xb_text = MathTex("\\vec{\\beta}\\vec{x}", color=GOLD).next_to(xb_vector.get_end(), direction=DOWN)
        xb = VGroup(xb_vector, xb_text)

        e_vector = Arrow(start=axes2.coords_to_point(9 * 0.5333, 3 * 0.5333), end=axes2.coords_to_point(3, 7), buff=-1,
                         color=GREEN)
        e_text = MathTex("\\vec{e}", color=GREEN).move_to(axes2.coords_to_point(4, 6))
        e = VGroup(e_vector, e_text)

        reg_line = Line(start=axes1.coords_to_point(0, 0), end=axes1.coords_to_point(10, 5.333), color=GREEN)

        theta_angle = Angle(x_vector, e_vector, radius=.75, other_angle=False)
        theta_text = MathTex("90^{\\circ}").move_to(axes2.coords_to_point(6, 3.4))
        theta = VGroup(theta_angle, theta_text)

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

        res = ([3 - 9 * 0.5333, 7 - 3 * 0.5333])

        MSE_text = MathTex("\\text{MSE} = ").move_to(axes1.coords_to_point(6, 10))
        MSE_val = DecimalNumber(np.mean([res[0] ** 2, res[1] ** 2])).next_to(MSE_text)
        MSE = VGroup(MSE_text, MSE_val)

        self.play(ReplacementTransform(VGroup(numberplane, coords), VGroup(axes1, axes2)))
        self.wait()
        self.play(GrowFromPoint(p1, point=axes1.coords_to_point(9, 3)),
                  GrowFromPoint(p2, point=axes1.coords_to_point(3, 7)),
                  Write(VGroup(y, x)))
        self.wait(5)

        regr = VGroup(reg_line, xb)
        self.play(Write(regr))
        self.wait()
        orthos = VGroup(right_angle_line1, right_angle_line2, e)
        self.play(Write(VGroup(orthos, MSE)))
        self.play(Write(theta))
        self.wait(5)

        ## too short

        # Create the "observations"
        p1_point = Dot(axes1.coords_to_point(9, 3), radius=0.1, color=WHITE)
        p1_text = MathTex("(y_1, x_1)", color=WHITE).next_to(p1_point, direction=DOWN)
        p1s = VGroup(p1_point, p1_text)

        p2_point = Dot(axes1.coords_to_point(3, 7), radius=0.1, color=WHITE)
        p2_text = MathTex("(y_2, x_2)", color=WHITE).next_to(p2_point, direction=UP)
        p2s = VGroup(p2_point, p2_text)

        x_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9, 3), buff=-1)
        x_text = MathTex("\\vec{x}").next_to(x_vector.get_end(), direction=DOWN)
        xs = VGroup(x_vector, x_text)

        y_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(3, 7), buff=-1)
        y_text = MathTex("\\vec{y}").next_to(y_vector.get_end(), direction=LEFT)
        ys = VGroup(y_vector, y_text)

        xb_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9 * 0.85, 3 * 0.85), buff=-1,
                          color=GOLD)
        xb_text = MathTex("\\vec{\\beta}\\vec{x}", color=GOLD).next_to(xb_vector.get_end(), direction=DOWN)
        xbs = VGroup(xb_vector, xb_text)

        e_vector = Arrow(start=axes2.coords_to_point(9 * 0.85, 3 * 0.85), end=axes2.coords_to_point(3, 7), buff=-1,
                         color=RED)
        e_text = MathTex("\\vec{e}", color=RED).move_to(axes2.coords_to_point(4.5, 6.5))
        es = VGroup(e_vector, e_text)

        reg_lines = Line(start=axes1.coords_to_point(0, 0), end=axes1.coords_to_point(10, 8.5), color=RED)

        theta_angle = Angle(x_vector, e_vector, radius=.75, other_angle=False)
        theta_text = MathTex(">90^{\\circ}").move_to(axes2.coords_to_point(8, 4.4))
        thetas = VGroup(theta_angle, theta_text)

        right_angle_point1, e1 = right_angler(A=[0, 0], B=[10, 8.5], C=[9, 3])
        right_angle_line1s = DashedLine(start=axes1.coords_to_point(9, 3),
                                       end=axes1.coords_to_point(right_angle_point1[0], right_angle_point1[1]),
                                       dash_length=0.1,
                                       color=TEAL)

        right_angle_point2, e2 = right_angler(A=[0, 0], B=[10, 8.5], C=[3, 7])
        right_angle_line2s = DashedLine(start=axes1.coords_to_point(3, 7),
                                       end=axes1.coords_to_point(right_angle_point2[0], right_angle_point2[1]),
                                       dash_length=0.1,
                                       color=TEAL)

        ress = ([3 - 9 * 0.8, 7 - 3 * 0.8])

        MSE_text = MathTex("\\text{MSE} = ").move_to(axes1.coords_to_point(6, 10))
        MSE_val = DecimalNumber(np.mean([ress[0] ** 2, ress[1] ** 2])).next_to(MSE_text)
        MSEs = VGroup(MSE_text, MSE_val)

        self.play(FadeOut(VGroup(orthos, theta)))
        regrs = VGroup(reg_lines, xbs)
        self.play(ReplacementTransform(regr, regrs))
        self.play(ReplacementTransform(MSE, MSEs))
        self.wait()
        orthoss = VGroup(right_angle_line1s, right_angle_line2s, es)
        self.play(Write(orthoss))
        self.play(Write(thetas))
        self.wait(5)

        ## too long

        # Create the "observations"
        p1_point = Dot(axes1.coords_to_point(9, 3), radius=0.1, color=WHITE)
        p1_text = MathTex("(y_1, x_1)", color=WHITE).next_to(p1_point, direction=DOWN)
        p1l = VGroup(p1_point, p1_text)

        p2_point = Dot(axes1.coords_to_point(3, 7), radius=0.1, color=WHITE)
        p2_text = MathTex("(y_2, x_2)", color=WHITE).next_to(p2_point, direction=UP)
        p2l = VGroup(p2_point, p2_text)

        x_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9, 3), buff=-1)
        x_text = MathTex("\\vec{x}").next_to(x_vector.get_end(), direction=DOWN)
        xl = VGroup(x_vector, x_text)

        y_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(3, 7), buff=-1)
        y_text = MathTex("\\vec{y}").next_to(y_vector.get_end(), direction=LEFT)
        yl = VGroup(y_vector, y_text)

        xb_vector = Arrow(start=axes2.coords_to_point(0, 0), end=axes2.coords_to_point(9 * 0.35, 3 * 0.35), buff=-1,
                          color=GOLD)
        xb_text = MathTex("\\vec{\\beta}\\vec{x}", color=GOLD).next_to(xb_vector.get_end(), direction=UL)
        xbl = VGroup(xb_vector, xb_text)

        e_vector = Arrow(start=axes2.coords_to_point(9 * 0.35, 3 * 0.35), end=axes2.coords_to_point(3, 7), buff=-1,
                         color=RED)
        e_text = MathTex("\\vec{e}", color=RED).move_to(axes2.coords_to_point(4, 6))
        el = VGroup(e_vector, e_text)

        reg_linel = Line(start=axes1.coords_to_point(0, 0), end=axes1.coords_to_point(10, 4), color=RED)

        theta_angle = Angle(x_vector, e_vector, radius=.75, other_angle=False)
        theta_text = MathTex("<90^{\\circ}").move_to(axes2.coords_to_point(5, 3))
        thetal = VGroup(theta_angle, theta_text)

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

        resl = ([3 - 9 * 0.4, 7 - 3 * 0.4])

        MSE_text = MathTex("\\text{MSE} = ").move_to(axes1.coords_to_point(6, 10))
        MSE_val = DecimalNumber(np.mean([resl[0] ** 2, resl[1] ** 2])).next_to(MSE_text)
        MSEl = VGroup(MSE_text, MSE_val)

        self.play(FadeOut(VGroup(orthoss, thetas)))
        regrl = VGroup(reg_linel, xbl)
        self.play(ReplacementTransform(regrs, regrl))
        self.play(ReplacementTransform(MSEs, MSEl))
        self.wait()
        orthol = VGroup(right_angle_line1l, right_angle_line2l, el)
        self.play(Write(orthol))
        self.play(Write(thetal))
        self.wait(2)
        self.wait(1)

