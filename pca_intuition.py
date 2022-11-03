# Execute in Terminal to render: manim -pql pca_intuition.py PCA_Analysis
import numpy as np
from manim import *
from sklearn.decomposition import PCA
from sklearn import preprocessing


# Set seed
np.random.seed(19052022)

# Create data set
V = np.array([[1, -0.1, 0.75], [-0.1, 1, 0.5], [0.75, 0.5, 1]])

dat = np.random.multivariate_normal([0, 0, 0], V, 200)

eigen = np.linalg.eig(np.matmul(dat.transpose(), dat)/(200 - 1))

PC = np.array(eigen[1] * np.sqrt(eigen[0] * (200 - 1)) / 2).transpose()
print(np.matmul(PC[0], PC[1]))
print(PC[0])

class PCA_Analysis(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)

        # Write intro and transform it cool
        intro = Text("Principal Component Analysis")
        self.add_fixed_orientation_mobjects(intro)
        self.wait()
        self.play(FadeOut(intro))
        self.wait()


        # Set seed
        np.random.seed(19052022)

        # Create data set
        #e = np.random.standard_normal([200, 3])
        #x1 = np.random.standard_normal(200)
        #x2 = np.random.standard_normal(200)
        #x3 = np.random.standard_normal(200)
        #X = np.c_[x1, x2, x3]

        #A = np.random.uniform(-0.9, 0.9, size=[3, 3])
        #A[0, 0] = 0
        #A[1, 1] = 0
        #A[2, 2] = 0

        #L = np.zeros([3, 3])
        #np.fill_diagonal(L, -np.sort(-np.random.uniform(0.1, 0.8, size=[3])))

        #Phi = np.matmul(A, L, A.transpose())

        #dat = np.matmul(Phi, X.transpose()).transpose() + e

        V = np.array([[1, -0.1, 0.75], [-0.1, 1, 0.5], [0.75, 0.5, 1]])

        dat = np.random.multivariate_normal([0, 0, 0], V, 200)

        axes = ThreeDAxes()
        data = VGroup(Dot3D(point=dat[0], color=WHITE))
        for t in range(1, 199):
            data = VGroup(data, Dot3D(point=dat[t], color=WHITE, radius=0.05))

        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.wait()
        self.play(GrowFromPoint(VGroup(data, axes), point=ORIGIN))

        #self.play(GrowFromPoint(VGroup(axes), point=ORIGIN))
        self.wait()

        eigen = np.linalg.eig(np.matmul(dat.transpose(), dat)/(200 - 1))

        PC = np.array(eigen[1] * np.sqrt(eigen[0] * (200 - 1)) / 2).transpose()

        PC1 = VGroup(Arrow(start=PC[0], end=ORIGIN, color=RED, stroke_width=3, stroke_opacity=0.5),
                     Arrow(start=ORIGIN, end=PC[0], color=RED, stroke_width=3, stroke_opacity=0.5)).move_to(np.mean(dat, 0))
        PC3 = VGroup(Arrow(start=PC[1], end=ORIGIN, color=RED, stroke_width=3, stroke_opacity=0.5),
                     Arrow(start=ORIGIN, end=PC[1], color=RED, stroke_width=3, stroke_opacity=0.5)).move_to(np.mean(dat, 0))
        PC2 = VGroup(Arrow(start=PC[2], end=ORIGIN, color=RED, stroke_width=3, stroke_opacity=0.5),
                     Arrow(start=ORIGIN, end=PC[2], color=RED, stroke_width=3, stroke_opacity=0.5)).move_to(np.mean(dat, 0))

        PC1_text = MathTex('\\text{PC}_1', color=RED).next_to(PC1[0].get_end())
        PC2_text = MathTex('\\text{PC}_2', color=RED).next_to(PC2[0].get_end())
        PC3_text = MathTex('\\text{PC}_3', color=RED).next_to(PC3[0].get_end())

        self.play(GrowFromPoint(PC1, np.mean(dat, 0)))
        self.wait()
        self.add_fixed_orientation_mobjects(PC1_text)
        self.wait()

        self.play(GrowFromPoint(PC2, np.mean(dat, 0)))
        self.wait()
        self.add_fixed_orientation_mobjects(PC2_text)
        self.wait()

        self.play(GrowFromPoint(PC3, np.mean(dat, 0)))
        self.wait()
        self.add_fixed_orientation_mobjects(PC3_text)
        self.wait()

        self.begin_ambient_camera_rotation(rate=1)
        self.wait(2*PI)
        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(axes,
                          data,
                          PC1, PC2, PC3, PC1_text, PC2_text, PC3_text))
        self.wait()
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.wait()

        # part 2
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.wait()

        self.play(GrowFromPoint(VGroup(data, axes), point=ORIGIN))
        # self.play(GrowFromPoint(VGroup(axes), point=ORIGIN))
        self.wait()

        space = Cube(side_length=1, fill_color=BLUE, stroke_width=1, fill_opacity=0.1)
        space.set_stroke(BLUE)

        e1 = Vector(np.array([1, 0, 0]), color=BLUE)
        e2 = Vector(np.array([0, 1, 0]), color=BLUE)
        e3 = Vector(np.array([0, 0, 1]), color=BLUE)

        PCe1 = Vector(-0.5*PC.transpose() @ np.array([1, 0, 0]), color=RED)
        PCe2 = Vector(-0.5*PC.transpose() @ np.array([0, 1, 0]), color=RED)
        PCe3 = Vector(-0.5*PC.transpose() @ np.array([0, 0, 1]), color=RED)

        self.play(GrowFromCenter(space),
                  GrowArrow(e1),
                  GrowArrow(e2),
                  GrowArrow(e3))
        self.wait()

        PC_space = ApplyMatrix(-1*PC.transpose(), space, color=RED)

        self.play(PC_space, ReplacementTransform(e1, PCe1), ReplacementTransform(e2, PCe2), ReplacementTransform(e3, PCe3))

        self.begin_ambient_camera_rotation(rate=1)
        self.wait(2 * PI)
        self.stop_ambient_camera_rotation()
        self.wait()