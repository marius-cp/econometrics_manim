# Execute in Terminal to render: python - m manim -pql manim/colinearity.py Colinearity
import configparser

import numpy as np
from manim import *

## Create data
np.random.seed(25082024)

N = 400

# The desired mean values of the sample.
mu = np.array([0.0, 0.0])

# The desired covariance matrix.
V = np.array([
        [  1, -0.75],
        [ -0.75,  1]
    ])

# Generate the random samples.
seed = np.random.default_rng()
X = np.array(seed.multivariate_normal(mu, V, size=N));

print(X)

config.disable_caching = False




class Colinearity(ThreeDScene, VectorScene):
    def construct(self):

        # Set default camara orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=60* DEGREES, zoom=0.7)
        axes = ThreeDAxes()

        # Create axes and corresponding labels
        lab_x = axes.get_x_axis_label(Tex("$x_1$"))
        lab_y = axes.get_y_axis_label(Tex("$x_2$"))
        lab_z = axes.get_z_axis_label(Tex("$y$"))
        coords = VGroup(axes, lab_x, lab_y, lab_z)

        intro = Text("Colinearity")
        self.add_fixed_orientation_mobjects(intro)
        self.wait()
        self.play(FadeOut(intro), Write(coords))
        self.wait()

        # Set axes for correlation indication
        axes2 = Axes(
            x_range=(-1.1, 1.1, 1),
            y_range=(-1.1, 1.1, 1),
            x_length=2,
            y_length=2,
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
                "include_ticks": False,
            },
            tips = False
        ).to_corner(UL)

        UC = Circle(1 * axes2.get_x_unit_size(), WHITE).move_to(axes2.get_origin())

        self.add_fixed_in_frame_mobjects(axes2)
        self.remove(axes2)
        self.play(Write(axes2))
        self.add_fixed_in_frame_mobjects(UC)
        self.remove(UC)
        self.play(Write(UC))
        self.wait()

        ## Orthogonal Data

        x_vec = Arrow(start=axes2.get_origin(), end=axes2.c2p(1, 0), color=RED, buff=0)
        y_vec = Arrow(start=axes2.get_origin(), end=axes2.c2p(0, 1), color=TEAL, buff=0)
        corr = MathTex("\\rho = 0.0").move_to(axes2.get_origin() + [2, 0, 0])

        self.add_fixed_in_frame_mobjects(x_vec)
        self.remove(x_vec)
        self.play(Write(x_vec))
        self.add_fixed_in_frame_mobjects(y_vec)
        self.remove(y_vec)
        self.play(Write(y_vec))
        self.add_fixed_in_frame_mobjects(corr)
        self.remove(corr)
        self.play(Write(corr))
        self.wait()

        # Create data
        np.random.seed(25082024)

        beta1 = 0.5
        beta2 = -0.75

        N = 300

        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0])

        # The desired covariance matrix.
        V = np.array([
            [1, 0],
            [0, 1]
        ])

        # Generate the random samples.
        seed = np.random.default_rng()
        X = np.array(seed.multivariate_normal(mu, V, size=N))
        e = np.random.normal(0, 1, N)

        y = np.matmul(X, np.array([beta1, beta1])) + e

        dat = np.column_stack((X, y))

        print(dat[0])

        data_list = []
        for t in range(0, N-1):
            data_list.append(Dot3D(np.array(dat[t]) * np.array([axes.get_x_unit_size(), axes.get_y_unit_size(), 1]),
                                   color=WHITE, radius=0.05, fill_opacity=0.5))

        data = VGroup(*data_list)
        self.play(Write(data))
        self.wait()

        self.begin_ambient_camera_rotation(rate=2)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()

        beta_hat = np.matmul(np.linalg.inv((np.matmul(X.transpose(), X))), np.matmul(X.transpose(), y))

        ## Draw true OLS plane
        def OLS_plane(u, v):
            x1 = u * axes.get_x_unit_size()
            x2 = v * axes.get_y_unit_size()
            beta_hat
            y_hat = beta_hat[0] * x1 + beta_hat[1] * x2
            return np.array([x1, x2, y_hat])

        predicted_plane_true = Surface(
            OLS_plane,
            v_range=[-5 * axes.get_x_unit_size(), +5 * axes.get_x_unit_size()],
            u_range=[-5 * axes.get_y_unit_size(), +5* axes.get_y_unit_size()]
        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        self.play(Write(predicted_plane_true))
        self.wait()

        self.play(FadeOut(data, predicted_plane_true))

        self.move_camera(phi=80 * DEGREES, theta=75 * DEGREES)
        self.wait()

        # slightly correlated data

        x_vec2 = Arrow(start=axes2.get_origin(), end=axes2.c2p(np.cos(np.pi/4), np.sin(np.pi/4)), color=RED, buff=0)
        self.add_fixed_in_frame_mobjects(x_vec2)
        self.remove(x_vec2)
        self.play(ReplacementTransform(x_vec, x_vec2))
        corr2 = MathTex("\\rho = 0.5").move_to(axes2.get_origin() + [2, 0, 0])
        self.add_fixed_in_frame_mobjects(corr2)
        self.remove(corr2)
        self.play(ReplacementTransform(corr, corr2))

        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0])

        # The desired covariance matrix.
        V = np.array([
            [1, 0.5],
            [0.5, 1]
        ])

        # Generate the random samples.
        seed = np.random.default_rng()
        X = np.array(seed.multivariate_normal(mu, V, size=N))
        e = np.random.normal(0, 1, N)

        y = np.matmul(X, np.array([beta1, beta1])) + e

        dat = np.column_stack((X, y))

        print(dat[0])

        data_list = []
        for t in range(0, N - 1):
            data_list.append(Dot3D(np.array(dat[t]) * np.array([axes.get_x_unit_size(), axes.get_y_unit_size(), 1]),
                                 color=WHITE, radius=0.05, fill_opacity=0.5))

        data = VGroup(*data_list)
        self.play(Write(data))
        self.wait()

        self.begin_ambient_camera_rotation(rate=2)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()

        beta_hat = np.matmul(np.linalg.inv((np.matmul(X.transpose(), X))), np.matmul(X.transpose(), y))

        ## Draw true OLS plane
        def OLS_plane(u, v):
            x1 = u * axes.get_x_unit_size()
            x2 = v * axes.get_y_unit_size()
            beta_hat
            y_hat = beta_hat[0] * x1 + beta_hat[1] * x2
            return np.array([x1, x2, y_hat])

        predicted_plane_true = Surface(
            OLS_plane,
            v_range=[-5 * axes.get_x_unit_size(), +5 * axes.get_x_unit_size()],
            u_range=[-5 * axes.get_y_unit_size(), +5 * axes.get_y_unit_size()]
        ).set_style(fill_opacity=0.2, fill_color=GREEN, stroke_opacity=0.2, stroke_color=GREEN)

        self.play(Write(predicted_plane_true))
        self.wait()

        self.play(FadeOut(data, predicted_plane_true))

        self.move_camera(phi=80 * DEGREES, theta=75 * DEGREES)
        self.wait()

        # colinear data

        x_vec3 = Arrow(start=axes2.get_origin(), end=axes2.c2p(0, 1), color=RED, buff=0)
        self.add_fixed_in_frame_mobjects(x_vec3)
        self.remove(x_vec3)
        self.play(ReplacementTransform(x_vec2, x_vec3))
        corr3 = MathTex("\\rho = 1.0").move_to(axes2.get_origin() + [2, 0, 0])
        self.add_fixed_in_frame_mobjects(corr3)
        self.remove(corr3)
        self.play(ReplacementTransform(corr2, corr3))

        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0])

        # The desired covariance matrix.
        V = np.array([
            [1, 1],
            [1, 1]
        ])

        # Generate the random samples.
        seed = np.random.default_rng()
        X = np.array(seed.multivariate_normal(mu, V, size=N))
        e = np.random.normal(0, 1, N)

        y = np.matmul(X, np.array([beta1, beta1])) + e

        dat = np.column_stack((X, y))

        print(dat[0])

        data_list = []
        for t in range(0, N - 1):
            data_list.append(Dot3D(np.array(dat[t]) * np.array([axes.get_x_unit_size(), axes.get_y_unit_size(), 1]),
                                   color=WHITE, radius=0.05, fill_opacity=0.5))

        data = VGroup(*data_list)
        self.play(Write(data))
        self.wait()

        self.begin_ambient_camera_rotation(rate=2)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()

        X1 = np.array(X.transpose()[0]).transpose()
        slope = 1/(np.matmul(X1.transpose(), X1)) * np.matmul(X1.transpose(), y)

        reg_lin = Line3D(start=[-5, -5, -5*slope], end=[5, 5, 5*slope], color=BLUE)

        def data_plane(u, v):
            x1 = u * axes.get_x_unit_size()
            x2 = u * axes.get_y_unit_size()
            y_hat = v
            return np.array([x1, x2, y_hat])

        data_plane_true = Surface(
            data_plane,
            v_range=[-5 * axes.get_x_unit_size(), +5 * axes.get_x_unit_size()],
            u_range=[-5 * axes.get_y_unit_size(), +5 * axes.get_y_unit_size()]
        ).set_style(fill_opacity=0.2, fill_color=RED, stroke_opacity=0.2, stroke_color=RED)

        self.play(Write(data_plane_true))
        self.wait()

        self.move_camera(phi=90 * DEGREES, theta=-45 * DEGREES)
        self.wait()
        self.play(Write(reg_lin))
        self.wait()
