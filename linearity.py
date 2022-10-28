# Execute in Terminal to render: python - m manim -pql linearity.py Linearity
# start 18:07
import configparser

import numpy as np
from manim import *

## Create data
np.random.seed(25082024)

N = 10

X = np.random.uniform(-5, 5, N)

y = np.transpose(0.5 * X.transpose() + 0.5 * np.random.standard_normal(N))

dat = np.column_stack((X, y))


## Derive OLS
beta_hat = (N * np.sum(np.multiply(X, y)) - np.sum(X)*np.sum(y))/(N*np.sum(X**2) - (np.sum(X))**2)

print(beta_hat)

config.disable_caching = True




class Linearity(ThreeDScene, VectorScene):
    def construct(self):

        #### Linearity #############################################################################

        #### Case 0: 2D ################################################################################################

        # Set default camara orientation
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=0.7)
        axes = ThreeDAxes()

        # Create axes and corresponding labels
        lab_x0 = axes.get_x_axis_label(Tex("$x_1$"))
        lab_y0 = axes.get_y_axis_label(Tex("$y$"))
        lab_z0 = axes.get_z_axis_label(Tex(""))
        coords0 = VGroup(axes, lab_x0, lab_y0)

        intro = Text("Linearity in the Coefficients")
        self.add(intro)
        self.wait()
        self.play(ReplacementTransform(intro, VGroup(coords0, lab_z0)))

        ## Create data
        np.random.seed(25082021)

        eq0 = MathTex("&y = \\beta_1x_1 + \epsilon\\\\",
                      "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\beta}_1x_1").to_corner(UP + LEFT).scale(
            0.5)
        self.add_fixed_in_frame_mobjects(eq0[0])
        self.remove(eq0[0])
        self.play(Write(eq0[0]))

        N = 50

        X = np.random.uniform(-5, 5, N)

        y = np.transpose(0.5 * X.transpose() + 0.5 * np.random.standard_normal(N))

        dat = np.column_stack((X, y, np.zeros(N)))

        data = VGroup(Dot3D(point=dat[0], color=WHITE, radius=0.05))
        for t in range(1, N - 1):
            data = VGroup(data, Dot3D(point=dat[t], color=WHITE, radius=0.05))

        self.play(Write(data))
        self.wait()

        ## Derive OLS
        beta_hat = (N * np.sum(np.multiply(X, y)) - np.sum(X)*np.sum(y))/(N*np.sum(X**2) - (np.sum(X))**2)

        predicted_line = ParametricFunction(lambda u: np.array([u, beta_hat * u, 0]), t_range=np.array([-5, 5]), fill_opacity=0.2, color=GREEN)

        lines = VGroup(
            Line3D(start=dat[0], end=[dat[0][0], beta_hat * dat[0][0], 0],
                   stroke_opacity=0.5, stroke_width=0.05))
        for t in range(1, N - 1):
            lines = VGroup(lines, VGroup(
                Line3D(start=dat[t], end=[dat[t][0], beta_hat * dat[t][0], 0],
                       stroke_opacity=0.5, stroke_width=0.05)))

        self.add_fixed_in_frame_mobjects(eq0[1])
        self.remove(eq0[1])
        self.play(Write(eq0[1]))
        self.wait()
        self.play(Write(VGroup(predicted_line, lines)))
        self.wait(5)

        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(data, predicted_line, eq0[0], eq0[1], lines))

        # First case: Simple linear data ###############################################################################

        # Create axes and corresponding labels
        lab_x = axes.get_x_axis_label(Tex("$x_1$"))
        lab_y = axes.get_y_axis_label(Tex("$x_2$"))
        lab_z = axes.get_z_axis_label(Tex("$y$"))
        coords = VGroup(axes, lab_x, lab_y)

        self.play(ReplacementTransform(VGroup(coords0, lab_z0), VGroup(coords, lab_z)))
        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES)
        self.wait()

        ## Create data
        np.random.seed(25082022)

        eq1 = MathTex("&y = \\beta_1x_1 + \\beta_2x_2 +  \epsilon\\\\",
                      "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(UP + LEFT).scale(0.5)
        self.add_fixed_in_frame_mobjects(eq1[0])
        self.remove(eq1[0])
        self.play(Write(eq1[0]))

        X = np.zeros([2, N]).transpose()
        X.transpose()[0] = np.random.uniform(-5, 5, N)
        X.transpose()[1] = np.random.uniform(-5, 5, N)

        y = np.transpose(0.5 * X.transpose()[0] - 0.25 * X.transpose()[1] + 0.5 * np.random.standard_normal(N))

        dat = np.column_stack((X, y))

        data = VGroup(Dot3D(point=dat[0], color=WHITE, radius=0.05))
        for t in range(1, N-1):
            data = VGroup(data, Dot3D(point=dat[t], color=WHITE, radius=0.05))

        self.play(Write(data))
        # self.play(GrowFromPoint(VGroup(axes), point=ORIGIN))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(5)

        ## Derive OLS
        beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)

        ## Draw OLS plane
        def OLS_plane(u, v):
            x1 = u
            x2 = v
            beta_hat
            y_hat = beta_hat[0]*x1 + beta_hat[1]*x2
            return np.array([x1, x2, y_hat])

        predicted_plane = Surface(
            OLS_plane,
            v_range=[-5, +5],
            u_range=[-5, +5]
        )

        predicted_plane.set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        lines = VGroup(Line3D(start=dat[0], end=[dat[0][0], dat[0][1], beta_hat[0]*dat[0][0] + beta_hat[1]*dat[0][1]],
                              stroke_opacity=0.5, stroke_width=0.05))
        for t in range(1, N - 1):
            lines = VGroup(lines, VGroup(Line3D(start=dat[t], end=[dat[t][0], dat[t][1], beta_hat[0]*dat[t][0] + beta_hat[1]*dat[t][1]],
                                                stroke_opacity=0.5, stroke_width=0.05)))

        self.add_fixed_in_frame_mobjects(eq1[1])
        self.remove(eq1[1])
        self.play(Write(eq1[1]))
        self.wait()
        self.play(Write(VGroup(predicted_plane, lines)))
        self.wait(5)

        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(data, predicted_plane, eq1[0], eq1[1], lines))
        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES)
        self.wait()

        # Second case: y is exponentially dependent ####################################################################

        eq2 = MathTex("&y = \\exp\\left(\\beta_1x_1 + \\beta_2x_2 +  \epsilon\\right)\\\\",
                      "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(UP + LEFT).scale(
            0.5)
        self.add_fixed_in_frame_mobjects(eq2[0])
        self.remove(eq2[0])
        self.play(Write(eq2[0]))

        ## Create data
        np.random.seed(25082023)

        X = np.zeros([2, N]).transpose()
        X.transpose()[0] = np.random.uniform(-5, 5, N)
        X.transpose()[1] = np.random.uniform(-5, 5, N)

        y = np.exp(np.transpose(0.5 * X.transpose()[0] - 0.25 * X.transpose()[1] + 0.5 * np.random.standard_normal(N)))

        dat = np.column_stack((X, y))

        data = VGroup(Dot3D(point=dat[0], color=WHITE, radius=0.05))
        for t in range(1, N - 1):
            data = VGroup(data, Dot3D(point=dat[t], color=WHITE, radius=0.05))

        self.play(Write(data))
        # self.play(GrowFromPoint(VGroup(axes), point=ORIGIN))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(5)

        ## Derive OLS
        beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)

        ## Draw OLS plane
        def OLS_plane(u, v):
            x1 = u
            x2 = v
            beta_hat
            y_hat = beta_hat[0] * x1 + beta_hat[1] * x2
            return np.array([x1, x2, y_hat])

        predicted_plane = Surface(
            OLS_plane,
            v_range=[-5, +5],
            u_range=[-5, +5]
        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        lines = VGroup(
            Line3D(start=dat[0], end=[dat[0][0], dat[0][1], beta_hat[0] * dat[0][0] + beta_hat[1] * dat[0][1]],
                   stroke_opacity=0.5, stroke_width=0.05))
        for t in range(1, N - 1):
            lines = VGroup(lines, VGroup(
                Line3D(start=dat[t], end=[dat[t][0], dat[t][1], beta_hat[0] * dat[t][0] + beta_hat[1] * dat[t][1]],
                       stroke_opacity=0.5, stroke_width=0.05)))

        self.add_fixed_in_frame_mobjects(eq2[1])
        self.remove(eq2[1])
        self.play(Write(eq2[1]))
        self.wait()
        self.play(Write(VGroup(predicted_plane, lines)))
        self.wait(5)

        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(predicted_plane, eq2[1], lines))
        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES)
        self.wait()

        ## Now after transformation
        eq2b = MathTex("&\\ln(y) = \\beta_1x_1 + \\beta_2x_2 +  \epsilon\\\\",
                       "&\\mathbb{E}(\\ln(y)|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(
            UP + LEFT).scale(0.5)
        lab_z2 = axes.get_z_axis_label(Tex("$\\ln y$"))

        ln_y = np.log(y)

        ln_dat = np.column_stack((X, ln_y))

        ln_data = VGroup(Dot3D(point=ln_dat[0], color=WHITE, radius=0.05))
        for t in range(1, N - 1):
            ln_data = VGroup(ln_data, Dot3D(point=ln_dat[t], color=WHITE, radius=0.05))

        ## Derive  true OLS
        beta_hat_true = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), ln_y)

        ## Draw true OLS plane
        def OLS_plane_true(u, v):
            x1 = u
            x2 = v
            beta_hat
            y_hat = beta_hat_true[0] * x1 + beta_hat_true[1] * x2
            return np.array([x1, x2, y_hat])

        predicted_plane_true = Surface(
            OLS_plane_true,
            v_range=[-5, +5],
            u_range=[-5, +5]
        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        lines = VGroup(
            Line3D(start=ln_dat[0], end=[ln_dat[0][0], ln_dat[0][1], beta_hat_true[0] * ln_dat[0][0] + beta_hat_true[1] * ln_dat[0][1]],
                   stroke_opacity=0.5, stroke_width=0.05))
        for t in range(1, N - 1):
            lines = VGroup(lines, VGroup(
                Line3D(start=ln_dat[t], end=[ln_dat[t][0], ln_dat[t][1], beta_hat_true[0] * ln_dat[t][0] + beta_hat_true[1] * ln_dat[t][1]],
                       stroke_opacity=0.5, stroke_width=0.05)))

        self.add_fixed_in_frame_mobjects(eq2b[0])
        self.remove(eq2b[0])
        self.play(ReplacementTransform(VGroup(lab_z, eq2[0]), VGroup(lab_z2, eq2b[0])))
        self.wait()
        self.play(ReplacementTransform(data, ln_data))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(5)

        ## Draw OLS plane again

        self.add_fixed_in_frame_mobjects(eq2b[1])
        self.remove(eq2b[1])
        self.play(Write(eq2b[1]))
        self.wait()
        self.play(Write(VGroup(predicted_plane_true, lines)))
        self.wait(5)

        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(predicted_plane_true, ln_data, eq2b[0], eq2b[1], lines))
        lab_z = axes.get_z_axis_label(Tex("$y$"))
        self.play(ReplacementTransform(lab_z2, lab_z))
        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES)
        self.wait()

        # Second case b: y is square root dependent (transformation) ###################################################

        eq3 = MathTex("&y = \\left(\\beta_1x_1 + \\beta_2x_2 +  \epsilon\\right)^2\\\\",
                      "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(UL).scale(
            0.5)
        self.add_fixed_in_frame_mobjects(eq3[0])
        self.remove(eq3[0])
        self.play(Write(eq3[0]))

        ## Create data
        np.random.seed(25082024)

        V = np.array([[2, 0.5], [0.5, 2]])

        X = np.zeros([2, N]).transpose()
        X.transpose()[0] = np.random.uniform(-5, 5, N)
        X.transpose()[1] = np.random.uniform(-5, 5, N)

        y = np.transpose(0.5 * X.transpose()[0] - 0.25 * X.transpose()[1] + 0.5 * np.random.standard_normal(N)) ** 2

        dat = np.column_stack((X, y))

        data = VGroup(Dot3D(point=dat[0], color=WHITE, radius=0.05))
        for t in range(1, N - 1):
            data = VGroup(data, Dot3D(point=dat[t], color=WHITE, radius=0.05))

        self.play(Write(data))
        # self.play(GrowFromPoint(VGroup(axes), point=ORIGIN))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait()

        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES)
        self.wait()

        ## Now after transformation
        lab_z3 = axes.get_z_axis_label(Tex("$\\sqrt{y}$"))
        eq3b = MathTex("&\\sqrt(y) = \\beta_1x_1 + \\beta_2x_2 +  \epsilon\\\\",
                       "&\\mathbb{E}(\\sqrt(y)|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(
            UP + LEFT).scale(0.5)

        y_sq = y ** 0.5

        sq_dat = np.column_stack((X, y_sq))

        sq_data = VGroup(Dot3D(point=sq_dat[0], color=WHITE, radius=0.05))
        for t in range(1, N - 1):
            sq_data = VGroup(sq_data, Dot3D(point=sq_dat[t], color=WHITE, radius=0.05))

        ## Derive  true OLS
        beta_hat_true = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y_sq)

        ## Draw true OLS plane
        def OLS_plane_true(u, v):
            x1 = u
            x2 = v
            beta_hat
            y_hat = beta_hat_true[0] * x1 + beta_hat_true[1] * x2
            return np.array([x1, x2, y_hat])

        predicted_plane_true = Surface(
            OLS_plane_true,
            v_range=[-5, +5],
            u_range=[-5, +5]
        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        lines = VGroup(
            Line3D(start=sq_dat[0],
                   end=[sq_dat[0][0], sq_dat[0][1], beta_hat_true[0] * sq_dat[0][0] + beta_hat_true[1] * sq_dat[0][1]],
                   stroke_opacity=0.5, stroke_width=0.05))
        for t in range(1, N - 1):
            lines = VGroup(lines, VGroup(
                Line3D(start=sq_dat[t],
                       end=[sq_dat[t][0], sq_dat[t][1],
                            beta_hat_true[0] * sq_dat[t][0] + beta_hat_true[1] * sq_dat[t][1]],
                       stroke_opacity=0.5, stroke_width=0.05)))

        self.add_fixed_in_frame_mobjects(eq3b[0])
        self.remove(eq3b[0])
        self.play(ReplacementTransform(VGroup(lab_z, eq3[0]), VGroup(lab_z3, eq3b[0])))
        self.wait()
        self.play(ReplacementTransform(data, sq_data))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(5)

        ## Draw OLS plane again

        self.add_fixed_in_frame_mobjects(eq3b[1])
        self.remove(eq3b[1])
        self.play(Write(eq3b[1]))
        self.wait()
        self.play(Write(VGroup(predicted_plane_true, lines)))
        self.wait(5)

        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(predicted_plane_true, eq3b[0], eq3b[1], lines), ReplacementTransform(sq_data, data))
        lab_z = axes.get_z_axis_label(Tex("$y$"))
        self.play(ReplacementTransform(lab_z3, lab_z))
        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES, frame_center=[-4, 0, 4])
        self.wait()

        ## Third case: y is squared dependent ##########################################################################

        eq4 = MathTex("&y = \\left(\\beta_1x_1 + \\beta_2x_2 +  \epsilon\\right)^2\\\\",
                      "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(UL).scale(
            0.5)
        self.add_fixed_in_frame_mobjects(eq4[0])
        self.remove(eq4[0])
        self.play(Write(eq4[0]))

        eq4b = MathTex("&y = \\beta_1^2x_1^2 + 2\\beta_1\\beta_2x_1x_2 +  \\beta_2^2x_2^2 + \\text{Rest}\\\\",
                       "&\\mathbb{E}(y|x_1,x_2)", "=", "\\text{const.}", "+", "\\hat{\\beta}_1^2x_1^2", "+",
                       "2\\hat{\\beta}_1\\hat{\\beta}_2x_1x_2", "+", "\\hat{\\beta}_2^2x_2^2\\\\",
                       "&\\hat{\\gamma}_0", ":=", "\\text{const.}\\\\",
                       "&\\hat{\\gamma}_1", ":=", "\\hat{\\beta}_1^2", "\\text{\\;and\\;}", "z_1", ":=", "x_1^2\\\\",
                       "&\\hat{\\gamma}_2", ":=", "2\\hat{\\beta}_1\\hat{\\beta}_2", "\\text{\\;and\\;}", "z_2", ":=",
                       "x_1x_2\\\\",
                       "&\\hat{\\gamma}_3", ":=", "\\hat{\\beta}_2^2", "\\text{\\;and\\;}", "z_3", ":=",
                       "x_2^2\\\\", ).to_corner(
            UL).scale(
            0.5)
        self.add_fixed_in_frame_mobjects(eq4b[0])
        self.remove(eq4b[0])
        self.play(ReplacementTransform(eq4[0], eq4b[0]))
        self.wait()
        self.add_fixed_in_frame_mobjects(eq4b[1:10])
        self.remove(eq4b[1:10])
        self.play(Write(eq4b[1:10]))
        self.wait(2)
        self.play(eq4b[3].animate.set_color(GREEN), eq4b[5].animate.set_color(BLUE),
                  eq4b[7].animate.set_color(YELLOW), eq4b[9].animate.set_color(RED))
        self.wait(1)

        self.add_fixed_in_frame_mobjects(eq4b[10:13])
        self.remove(eq4b[10:13])
        eq4b[10].set_color(GREEN)
        eq4b[12].set_color(GREEN)
        self.play(Write(eq4b[10:13]))
        self.wait(2)

        self.add_fixed_in_frame_mobjects(eq4b[13:20])
        self.remove(eq4b[13:20])
        eq4b[13].set_color(BLUE)
        eq4b[15].set_color(BLUE)
        eq4b[17].set_color(BLUE)
        eq4b[19].set_color(BLUE)
        self.play(Write(eq4b[13:20]))
        self.wait(2)

        self.add_fixed_in_frame_mobjects(eq4b[20:27])
        self.remove(eq4b[20:27])
        eq4b[20].set_color(YELLOW)
        eq4b[22].set_color(YELLOW)
        eq4b[24].set_color(YELLOW)
        eq4b[26].set_color(YELLOW)
        self.play(Write(eq4b[20:27]))
        self.wait(2)

        self.add_fixed_in_frame_mobjects(eq4b[27:34])
        self.remove(eq4b[27:34])
        eq4b[27].set_color(RED)
        eq4b[29].set_color(RED)
        eq4b[31].set_color(RED)
        eq4b[33].set_color(RED)
        self.play(Write(eq4b[27:34]))
        self.wait(2)

        eq4c = MathTex("&y = \\beta_1^2x_1^2 + 2\\beta_1\\beta_2x_1x_2 +  \\beta_2^2x_2^2 + \\text{Rest}\\\\",
                       "&\\mathbb{E}(y|x_1,x_2)", "=", "\\hat{\\gamma_0}", "+", "\\hat{\\gamma_1}z_1", "+",
                       "\\hat{\\gamma_2}z_2", "+", "\\hat{\\gamma_3}z_3").to_corner(
            UL).scale(
            0.5)
        eq4c[3].set_color(GREEN)
        eq4c[5].set_color(BLUE)
        eq4c[7].set_color(YELLOW)
        eq4c[9].set_color(RED)
        self.add_fixed_in_frame_mobjects(eq4c)
        self.remove(eq4c)
        self.play(ReplacementTransform(eq4b, eq4c))
        self.wait(2)
        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES, frame_center=ORIGIN)

        ## Multiplying out the model

        X1_sq = X.transpose()[0] ** 2
        X2_sq = X.transpose()[1] ** 2
        XX = X.transpose()[0] * X.transpose()[1]
        int = np.ones(N)
        Z = np.column_stack((int, X1_sq, XX, X2_sq))

        ## Derive  true OLS
        beta_hat_true = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.transpose(), Z)), Z.transpose()), y)

        ## Draw true OLS plane
        def OLS_plane_true(u, v):
            x1 = u
            x2 = v
            beta_hat
            y_hat = beta_hat_true[0] + beta_hat_true[1] * x1 ** 2 + beta_hat_true[2] * x2 * x1 + beta_hat_true[
                3] * x2 ** 2
            return np.array([x1, x2, y_hat])

        predicted_plane_true = Surface(
            OLS_plane_true,
            v_range=[-5, +5],
            u_range=[-5, +5]
        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        lines = VGroup(
            Line3D(start=dat[0],
                   end=[dat[0][0], dat[0][1],
                        beta_hat_true[0] + beta_hat_true[1] * dat[0][0] ** 2 + beta_hat_true[2] * dat[0][0] * dat[t][
                            1] + beta_hat_true[
                            3] * dat[0][1] ** 2],
                   stroke_opacity=0.5, stroke_width=0.05))
        for t in range(1, N - 1):
            lines = VGroup(lines, VGroup(
                Line3D(start=dat[t],
                       end=[dat[t][0], dat[t][1],
                            beta_hat_true[0] + beta_hat_true[1] * dat[t][0] ** 2 + beta_hat_true[2] * dat[t][0] *
                            dat[t][1] + beta_hat_true[3] * dat[t][1] ** 2],
                       stroke_opacity=0.5, stroke_width=0.05)))

        self.play(Write(VGroup(predicted_plane_true, lines)))
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.25)
        self.wait(5)

        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(
            VGroup(predicted_plane_true, data, eq4b[0], eq4c[1], lines, axes, lab_y, lab_z3, lab_x, lab_z, eq4c)))
        self.wait()

        ### Fourth case: y is a non-constructable function ###############################################################
#
#        eq5 = MathTex("&y = \\exp(\\beta_1x_1) * \\ln(\\beta_2x_2) +  \epsilon\\right)^2\\\\",
#                      "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\beta}_1x_1 + \\hat{\\beta}_2x_2").to_corner(UL).scale(
#            0.5)
#        self.add_fixed_in_frame_mobjects(eq5[0])
#        self.remove(eq5[0])
#        self.play(Write(eq5[0]))
#
#        # Create data
#        np.random.seed(25082025)
#
#        axes2 = ThreeDAxes(x_range=[-0.5, 0.5, 0.2],
#                           y_range=[-0.5, 0.5, 0.2])
#        # Transform axes axes and corresponding labels
#        lab_x2 = axes.get_x_axis_label(Tex("$x_1$"))
#        lab_y2 = axes.get_y_axis_label(Tex("$x_2$"))
#        coords2 = VGroup(axes2, lab_x2, lab_y2)
#        self.play(ReplacementTransform(coords, coords2))
#        self.wait()
#
#        X = np.zeros([2, N]).transpose()
#        X.transpose()[0] = np.random.uniform(-0.5, 0.5, N)
#        X.transpose()[1] = np.random.uniform(-0.5, 0.5, N)
#
#        y = np.transpose(
#            np.exp(0.5 * X.transpose()[0]) * np.log(-0.25 * X.transpose()[1] + 1) + np.random.standard_normal(N))
#
#        dat = np.matmul(np.column_stack((X, y)), np.array([[10 * axes.get_x_unit_size(), 0, 0],
#                                                          [0, 10 * axes.get_y_unit_size(), 0],
#                                                          [0, 0, 10]]))
#
#        data = VGroup(Dot3D(point=dat[0], color=WHITE, radius=0.05))
#        for t in range(1, N - 1):
#            data = VGroup(data, Dot3D(point=dat[t], color=WHITE, radius=0.05))
#
#        self.play(Write(data))
#        # self.play(GrowFromPoint(VGroup(axes), point=ORIGIN))
#        self.wait()
#
#       ## Approximating the Model
#
#        eq5b = MathTex("&y = \\gamma_1x_2 + \\gamma_2x_2^2 +  \\gamma_2x_2x_1 + \\epsilon\\\\",
#                       "&\\mathbb{E}(y|x_1,x_2) = \\text{const.} + \\beta_1^2x_1^2 + 2\\beta_1\\beta_2x_1x_2 +  \\beta_2^2x_2^2").to_corner(
#            UL).scale(
#            0.5)
#        self.add_fixed_in_frame_mobjects(eq5b[0])
#        self.remove(eq5b[0])
#        self.play(ReplacementTransform(eq5[0], eq5b[0]))
#        self.wait()
#        self.add_fixed_in_frame_mobjects(eq5b[1])
#        self.remove(eq5b[1])
#        self.play(Write(eq5b[1]))
#
#        eq5c = MathTex("&y = \\gamma_1x_2 + \\gamma_2x_2^2 +  \\gamma_2x_2x_1 + \\epsilon\\\\",
#                       "&\\mathbb{E}(y|x_1,x_2) = \\hat{\\gamma_1}z_1 + \\hat{\\gamma_2}z_2 +  \\hat{\\gamma_3}z_3").to_corner(
#            UL).scale(
#            0.5)
#        self.add_fixed_in_frame_mobjects(eq5c[1])
#        self.remove(eq5c[1])
#        self.play(ReplacementTransform(eq5b[1], eq5c[1]))
#        self.wait()
#
#        X2 = X.transpose()[1]
#        X2_sq = X.transpose()[1] ** 2
#        XX = X.transpose()[0] * X.transpose()[1]
#        Z = np.column_stack((XX, X2_sq, X2))
#
#        ## Derive  true OLS
#        beta_hat_true = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.transpose(), Z)), Z.transpose()), y)
#
#        ## Draw true OLS plane
#        def OLS_plane_true(u, v):
#            x1 = u
#            x2 = v
#            beta_hat
#            y_hat = beta_hat_true[0] * x1 * x2 + beta_hat_true[1] * x2 ** 2 + beta_hat_true[2] * x2
#            return np.array([x1 * 10, x2 * 10, y_hat * 10])
#
#        predicted_plane_true = Surface(
#            OLS_plane_true,
#            v_range=[-0.5, +0.5],
#            u_range=[-0.5, +0.5]
#        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)
#
#        lines = VGroup(
#            Line3D(start=dat[0],
#                   end=[dat[0][0], dat[0][1],
#                        beta_hat_true[0] * dat[0][0] * dat[0][1] + beta_hat_true[1] * dat[0][1] ** 2 + beta_hat_true[
#                            2] * dat[t][1]],
#                   stroke_opacity=0.5, stroke_width=0.05))
#        for t in range(1, N - 1):
#            lines = VGroup(lines, VGroup(
#                Line3D(start=dat[t],
#                       end=[dat[t][0], dat[t][1],
#                            beta_hat_true[0] * dat[t][0] * dat[t][1] + beta_hat_true[1] * dat[t][1] ** 2 + beta_hat_true[2] * dat[t][1]],
#                       stroke_opacity=0.5, stroke_width=0.05)))
#
#        self.play(Write(VGroup(predicted_plane_true, lines)))
#        self.wait()
#
#        self.begin_ambient_camera_rotation(rate=0.25)
#        self.wait(5)
#
#        self.stop_ambient_camera_rotation()
#        self.wait()
#
#        self.play(FadeOut(VGroup(predicted_plane_true, data, lines)))
#        lab_z = axes.get_z_axis_label(Tex("$y$"))
#        self.play(ReplacementTransform(lab_z3, lab_z))
#        self.move_camera(phi=80 * DEGREES, theta =-100 * DEGREES)
#        self.wait()
#








