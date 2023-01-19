# Execute in Terminal to render: python -m manim -pql manim/random_sampling_slope.py Random_Sampling_Slope
import numpy as np
from manim import *
import numpy as np
import random as rnd

np.random.seed(154112022)

N=300
beta = 0.7
delta = 0.5

X = np.random.normal(0, 5, N)
e = np.random.normal(0, 1.5, N)

rnd_ind = np.hstack(([1, 0], np.round(np.random.uniform(0.0001, 1, N - 2))))

y = beta * X + delta * rnd_ind + e
X_mat = np.column_stack((np.ones(N), X))
beta_hat = (np.matmul(np.matmul(np.linalg.inv(np.matmul(X_mat.transpose(), X_mat)), X_mat.transpose()), y))

sampling = np.sort(np.array(rnd.sample(range(0, N - 1), k=150)))
sample1 = np.array([x for x in sampling[sampling < 149]])
sample0 = np.array([x - 100 for x in sampling[150 <= sampling]])

print(sample0)
print(sample1)

class Random_Sampling_Slope(MovingCameraScene):
    def construct(self):

        np.random.seed(4112022)
        N = 300
        beta = 0.7
        delta = 2

        # Set axes
        axes = Axes(
            x_range=(-0.5, 10, 1),
            y_range=(-0.5, 10, 1),
            x_length=6,
            axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 2,
            }
        ).to_edge(LEFT).add_coordinates()
        # Create axes and corresponding labels
        labels = axes.get_axis_labels(
            x_label=Tex("$x$"), y_label=Tex("$y$")
        )

        # Set axes
        axes2 = Axes(
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
        ).to_edge(RIGHT).add_coordinates()
        # Create axes and corresponding labels
        labels2 = axes2.get_axis_labels(
            x_label=Tex("$x$"), y_label=Tex("$y$")
        )


        # Write intro and transform it cool
        intro = Text("Random Sampling: Slope Effects")
        self.add(intro)
        self.wait()

        labelings = VGroup(axes, labels)
        labelings2 = VGroup(axes2, labels2)

        self.play(ReplacementTransform(intro, VGroup(labelings, labelings2)),
                  self.camera.frame.animate.scale(1.5))
        ##### Different slopes instead of different intercepts

        np.random.seed(4112022)
        N = 300
        beta = 0.7
        beta1 = 0.4

        # Screen text
        text1 = Tex("``True'' Model: $y_n = \\gamma_0 + \\gamma_1 x_n + \\gamma_2 d_n + \\epsilon$, where $d_n = 1$ if"
                    "$i \in \mathcal{I}$.").move_to([0, 5, 0]).scale(0.75)
        text2 = Tex("``Assumed'' Model: $y_n = \\beta_0 + \\beta_1 x_n + \\epsilon$").move_to([0, 4.5, 0]).scale(0.75)
        text3 = Tex(
            "Note that, $\\beta_0 = \\gamma_0 + \\frac{1}{\\text{card}(\mathcal{I})}\\gamma_2$ and $\\beta_1 = \\gamma_1$.").move_to(
            [0, 4, 0]).scale(0.75)

        axes_label = Tex("Random Sampling").move_to(axes.c2p(5, 10)).scale(0.75)
        axes_label2 = Tex("Non-Random Sampling").move_to(axes2.c2p(5, 10)).scale(0.75)

        text4 = Tex("``True'' Model: $y_n = \\gamma_0 + \\gamma_1 x_n + \\gamma_2 x_n d_n + \\epsilon$, where $d_n = 1$ if"
                    "$i \in \mathcal{I}$.").move_to(
            [0, 5, 0]).scale(0.75)
        text5 = Tex("``Assumed'' Model: $y_n = \\beta_0 + \\beta_1 x_n + \\epsilon$").move_to([0, 4.5, 0]).scale(0.75)
        text6 = Tex(
            "Note that, $\\beta_0 = \\gamma_0$ and $\\beta_1 = \\gamma_1 + \\frac{1}{\\text{card}(\mathcal{I})}\\gamma_2$.").move_to(
            [0, 4, 0]).scale(0.75)


        self.play(Write(text4), Write(text5), Write(text6))
        self.wait()
        self.play(Write(axes_label))
        self.wait()
        self.play(Write(axes_label2))
        
        # Draw Data

        X = np.random.normal(7, 3, N) * axes.get_x_unit_size()
        e = np.random.normal(0, 1, N) * axes.get_x_unit_size()

        rnd_ind1 = np.hstack((np.zeros(int(N / 2)), np.ones(int(N / 2))))
        rnd_ind0 = np.hstack((np.ones(int(N / 2)), np.zeros(int(N / 2))))

        intercept = 1

        y = intercept + beta * X + beta1 * X * rnd_ind1 + e

        # data_1 = VGroup()
        # data_0 = VGroup()
        data_list = []
        data_list2 = []




        for t in range(0, N - 1):
            data_list.append(Dot(color=WHITE, radius=0.05).move_to(axes.c2p(X[t], y[t])))
            data_list2.append(Dot(color=WHITE, radius=0.05).move_to(axes2.c2p(X[t], y[t])))
            # if rnd_ind[t] == 1:
            #    data_1 = VGroup(data_1, VGroup(Dot(color=WHITE, radius=0.05).move_to(axes.c2p(X[t], y[t]))))
            # else:
            #    data_0 = VGroup(data_0, VGroup(Dot(color=WHITE, radius=0.05).move_to(axes.c2p(X[t], y[t]))))
        data = VGroup(*data_list)
        data2 = VGroup(*data_list2)

        self.play(Write(VGroup(data, data2)))
        self.wait(2)

        # colour the data

        self.play(data[0:(int(N / 2) - 1)].animate.set_color(BLUE),
                  data[(int(N / 2)):(N - 1)].animate.set_color(YELLOW),
                  data2[0:(int(N / 2) - 1)].animate.set_color(BLUE),
                  data2[(int(N / 2)):(N - 1)].animate.set_color(YELLOW))
        self.wait(2)

        # True regression line

        X_mat = np.column_stack((np.ones(N), X))
        beta_hat = (np.matmul(np.matmul(np.linalg.inv(np.matmul(X_mat.transpose(), X_mat)), X_mat.transpose()), y))
        beta_0_hat = beta_hat[0]
        beta_1_hat = beta_hat[1]

        reg_lin1 = Line(start=axes.c2p(-0.5, intercept - (beta + beta1) * 0.5),
                        end=axes.c2p(10, intercept + (beta + beta1) * 10),
                        color=YELLOW)

        reg_lin12 = Line(start=axes2.c2p(-0.5, intercept - (beta + beta1) * 0.5),
                         end=axes2.c2p(10, intercept + (beta + beta1) * 10),
                         color=YELLOW)

        self.play(Write(reg_lin1), Write(reg_lin12))

        reg_lin0 = Line(start=axes.c2p(-0.5, intercept - (beta) * 0.5), end=axes.c2p(10, intercept + (beta) * 10),
                        color=BLUE)
        reg_lin02 = Line(start=axes2.c2p(-0.5, intercept - (beta) * 0.5), end=axes2.c2p(10, intercept + (beta) * 10),
                         color=BLUE)
        self.play(Write(reg_lin0), Write(reg_lin02))
        self.wait(2)

        gen_reg_lin = Line(start=axes.c2p(-0.5, beta_0_hat - beta_1_hat * 0.5),
                           end=axes.c2p(10, beta_0_hat + beta_1_hat * 10), color=GOLD, stroke_width=5)
        gen_reg_lin2 = Line(start=axes2.c2p(-0.5, beta_0_hat - beta_1_hat * 0.5),
                            end=axes2.c2p(10, beta_0_hat + beta_1_hat * 10), color=GOLD, stroke_width=5)

        beta_1 = MathTex("\\beta_1 = ", np.round(beta_1_hat, 2)).set_color(GOLD).move_to(axes.c2p(2, 9)).scale(0.75)
        beta_12 = MathTex("\\beta_1 = ", np.round(beta_1_hat, 2)).set_color(GOLD).move_to(axes2.c2p(2, 9)).scale(0.75)
        self.play(data.animate.set_color(WHITE), ReplacementTransform(VGroup(reg_lin1, reg_lin0), gen_reg_lin),
                  Write(beta_1),
                  data2.animate.set_color(WHITE), ReplacementTransform(VGroup(reg_lin12, reg_lin02), gen_reg_lin2),
                  Write(beta_12)
                  )

        # choose randomly

        I = 20

        chosen = np.zeros([int(N / 2), I])
        chosen2 = np.zeros([int(N / 2), I])

        # resampling

        self.play(data.animate.set_color(DARKER_GREY), data2.animate.set_color(DARKER_GREY))
        self.wait(2)

        reg_lin_i_list = []
        beta_hat_i = []
        reg_lin_i_list2 = []
        beta_hat_i2 = []
        for i in range(0, I):
            chosen[:, i] = np.sort(np.array(rnd.sample(range(0, N - 1), k=int(N / 2))))
            X_i = np.column_stack((np.ones(int(N / 2)), X[chosen[:, i].astype(int)]))
            y_i = y[chosen[:, i].astype(int)]

            beta_i = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_i.transpose(), X_i)), X_i.transpose()), y_i)

            beta_hat_i.append(
                MathTex("\\hat{\\beta}_0 = ", np.round(beta_i[0], 3)).set_color(RED).move_to(axes.c2p(9, 1.5)).scale(
                    0.75))

            reg_lin_i_list.append(Line(start=axes.c2p(-0.5, beta_i[0] - beta_i[1] * 0.5),
                                       end=axes.c2p(10, beta_i[0] + beta_i[1] * 10), color=RED))

            # non-ranomd sampling

            chosen2[:, i] = np.hstack(
                (np.sort(np.array(rnd.sample(range(0, (int(N / 2) - 1)), k=(int(4 / 6 * N / 2))))),
                 np.sort(np.array(rnd.sample(range(int(N / 2), (N - 1)), k=int(2 / 6 * N / 2))))))
            X_i2 = np.column_stack((np.ones(int(N / 2)), X[chosen2[:, i].astype(int)]))
            y_i2 = y[chosen2[:, i].astype(int)]

            beta_i2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_i2.transpose(), X_i2)), X_i2.transpose()), y_i2)

            beta_hat_i2.append(
                MathTex("\\hat{\\beta}_0 = ", np.round(beta_i2[0], 3)).set_color(RED).move_to(axes2.c2p(9, 1.5)).scale(
                    0.75))

            reg_lin_i_list2.append(Line(start=axes2.c2p(-0.5, beta_i2[0] - beta_i2[1] * 0.5),
                                        end=axes2.c2p(10, beta_i2[0] + beta_i2[1] * 10), color=RED))

        reg_lin_i = VGroup(*reg_lin_i_list)
        reg_lin_i2 = VGroup(*reg_lin_i_list2)

        for i in range(0, 5):
            sampling = np.sort(chosen[:, i])
            sample1 = np.array(sampling[sampling < (int(N / 2) - 1)]).astype(int)
            sample0 = np.array(sampling[int(N / 2) <= sampling]).astype(int)

            current_draw1 = []
            current_draw0 = []

            for j in sample1:
                current_draw1.append(data[j])
            for j in sample0:
                current_draw0.append(data[j])

            cd1 = VGroup(*current_draw1)
            cd0 = VGroup(*current_draw0)

            # non-ranodmly sampled

            sampling2 = np.sort(chosen2[:, i])
            sample12 = np.array(sampling2[sampling2 < (int(N / 2) - 1)]).astype(int)
            sample02 = np.array(sampling2[int(N / 2) <= sampling2]).astype(int)

            current_draw12 = []
            current_draw02 = []

            for j in sample12:
                current_draw12.append(data2[j])
            for j in sample02:
                current_draw02.append(data2[j])

            cd12 = VGroup(*current_draw12)
            cd02 = VGroup(*current_draw02)

            self.play(AnimationGroup(*[_.animate.set_color(BLUE) for _ in cd1]),
                      AnimationGroup(*[_.animate.set_color(YELLOW) for _ in cd0]),
                      AnimationGroup(*[_.animate.set_color(BLUE) for _ in cd12]),
                      AnimationGroup(*[_.animate.set_color(YELLOW) for _ in cd02]))
            if (i == 0):
                self.play(Write(beta_hat_i[i], run_time=0.5), Write(reg_lin_i[i], run_time=0.05),
                          Write(beta_hat_i2[i], run_time=0.5), Write(reg_lin_i2[i], run_time=0.05))
            else:
                self.play(ReplacementTransform(beta_hat_i[i - 1], beta_hat_i[i]), Write(reg_lin_i[i], run_time=0.05),
                          ReplacementTransform(beta_hat_i2[i - 1], beta_hat_i2[i]), Write(reg_lin_i2[i], run_time=0.05))

            self.play(reg_lin_i[i].animate.set_color(GREY), data.animate.set_color(DARKER_GREY),
                      reg_lin_i2[i].animate.set_color(GREY), data2.animate.set_color(DARKER_GREY))
            self.bring_to_back(data, data2)

        for i in range(5, I):
            sampling = np.sort(chosen[:, i])
            sample1 = np.array(sampling[sampling < (int(N / 2) - 1)]).astype(int)
            sample0 = np.array(sampling[int(N / 2) <= sampling]).astype(int)

            current_draw1 = []
            current_draw0 = []

            for j in sample1:
                current_draw1.append(data[j])
            for j in sample0:
                current_draw0.append(data[j])

            cd1 = VGroup(*current_draw1)
            cd0 = VGroup(*current_draw0)

            # non-ranodm sampling

            sampling2 = np.sort(chosen2[:, i])
            sample12 = np.array(sampling2[sampling2 < (int(N / 2) - 1)]).astype(int)
            sample02 = np.array(sampling2[int(N / 2) <= sampling2]).astype(int)

            current_draw12 = []
            current_draw02 = []

            for j in sample12:
                current_draw12.append(data2[j])
            for j in sample02:
                current_draw02.append(data2[j])

            cd12 = VGroup(*current_draw12)
            cd02 = VGroup(*current_draw02)

            self.play(AnimationGroup(*[_.animate.set_color(BLUE) for _ in cd1]),
                      AnimationGroup(*[_.animate.set_color(YELLOW) for _ in cd0]),
                      AnimationGroup(*[_.animate.set_color(BLUE) for _ in cd12]),
                      AnimationGroup(*[_.animate.set_color(YELLOW) for _ in cd02]))

            self.play(ReplacementTransform(beta_hat_i[i - 1], beta_hat_i[i]), Write(reg_lin_i[i]),
                      ReplacementTransform(beta_hat_i2[i - 1], beta_hat_i2[i]), Write(reg_lin_i2[i]))

            self.play(reg_lin_i[i].animate.set_color(GREY), data.animate.set_color(DARKER_GREY),
                      reg_lin_i2[i].animate.set_color(GREY), data2.animate.set_color(DARKER_GREY))
            self.bring_to_back(data, data2)

        self.bring_to_front(gen_reg_lin, gen_reg_lin2)
        self.play(FadeOut(reg_lin_i, gen_reg_lin, beta_hat_i[I - 1], reg_lin_i2, gen_reg_lin2, beta_hat_i2[I - 1]))
        self.wait()