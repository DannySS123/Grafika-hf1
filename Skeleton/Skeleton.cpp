//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szaraz Daniel
// Neptun : GT5X34
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
		
	void main() {
		vec4 cp =  vec4(vp.x, vp.y, 0, 1) * MVP;
		float w = sqrt(cp.x * cp.x + cp.y * cp.y + 1);
		float t = 1/(w+1);
		gl_Position = vec4(t*cp.x, t*cp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

//CONSTANTS
const int nv = 100; //circel pieces
const int numOflinePoints = 100; //line pirces
const double hidrogenWeigth = 1.00797;
const double electronCharge = -1.602176634;
const double eps0 = 8.854187817;

class Camera2D {
	vec2 wCenter;
	vec2 wSize;
public:
	Camera2D() : wCenter(0, 0), wSize(600, 600) {}

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;
GPUProgram gpuProgram;

class Atom {
	vec2 pos, vel, mol_center;
	vec3 color;
	float charge, r;

	unsigned int vao;
	float sx, sy;
	vec2 wTranslate = vec2(0,0);
	float phi = 0;

public:
	Atom() { Animate(vec2(0, 0), 0.0); }

	void create(vec2 t, vec3 c, float s, float ch) {
		wTranslate = 0;		
		pos = t;
		color = c;
		sx = s;
		sy = s;
		r = s;
		charge = ch;

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		vec2 vertices[nv];
		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vertices[i] = vec2(cos(fi), sin(fi));
		}
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vec2) * nv,
			&vertices,
			GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	vec2 getPos() {
		return pos + wTranslate;
	}

	float getR() {
		return r;
	}

	float getCharge() {
		return charge;
	}

	void setMolCenter(vec2 c) {
		mol_center = c;
	}

	void Animate(vec2 r, float omega) {
		wTranslate = wTranslate + r;
		phi += omega;
	}

	mat4 M() {
		mat4 Mscale(
			sx, 0,  0, 0,
			0,  sy, 0, 0,
			0,  0,  1, 0,
			0,  0,  0, 1);

		mat4 MtranslateBefor(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			pos.x -mol_center.x, pos.y-mol_center.y, 0, 1);
		
		mat4 Mrotate(
			cosf(phi),	 sinf(phi), 0, 0,
			-sinf(phi),	 cosf(phi), 0, 0,
			0,			 0,			1, 0,
			0,			 0,			0, 1);

		mat4 Mtranslate(
			1,			  0,			0, 0,
			0,			  1,			0, 0,
			0,			  0,			1, 0,
			mol_center.x + wTranslate.x, mol_center.y + wTranslate.y,	0, 1);

		return Mscale * MtranslateBefor * Mrotate * Mtranslate;
	}

	void Draw() {
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
	}
};

class Line {
	vec2 start, end, middle, mol_center;
	vec3 color;

	unsigned int vao;
	float sx, sy;
	vec2 wTranslate = vec2(0,0);
	float phi = 0;;

public:
	Line() { Animate(vec2(0, 0), 0.0); }

	void create(vec2 start, vec2 end, vec3 c) {
		color = c;
		middle = start + ((end - start) / 2);
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		vec2 vertices[numOflinePoints];
		vertices[0] = start;
		for (int i = 1; i < numOflinePoints - 1; i++) {
			vertices[i] = vec2(start + (i*(end-start)/numOflinePoints));			
		}
		vertices[numOflinePoints - 1] = end;

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vec2) * numOflinePoints,
			&vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	void Animate(vec2 r, float omega) {
		sx = 1;
		sy = 1;
		wTranslate = wTranslate + r;
		phi += omega;
	}

	mat4 M() {
		mat4 Mscale(
			sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);

		mat4 MtranslateBefor(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			-mol_center.x, -mol_center.y, 0, 1);

		mat4 Mrotate(
			cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 Mtranslate(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			mol_center.x + wTranslate.x, mol_center.y + wTranslate.y, 0, 1);

		return Mscale * MtranslateBefor * Mrotate * Mtranslate;
	}

	void Draw() {
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z);

		glBindVertexArray(vao);
		glDrawArrays(GL_LINE_STRIP, 0, numOflinePoints);
	}

	void setMolCenter(vec2 c) {
		mol_center = c;
	}
};

float distance(vec2 a1, vec2 a2) {
	return length(a1 - a2);
}

class Molekule {
public:
	Atom atoms[8];
	Line lines[8];
	int numOfatoms;
	vec2 vel, center;
	float sumOfMass, omega, theta;
	vec2 rVec[8];
	vec3 sumM;
	vec2 sumF;

	Molekule() { 
		srand(6);
		numOfatoms = 0; 
		sumOfMass = 0;
		omega = 0;
		theta = 1;
	}

	void create()  {		
		double chargeSum = 0;
		sumOfMass = 0;
		numOfatoms = rand() % 7 + 2;
		float firstX = rand() % 501 - 250;
		float firstY = rand() % 501 - 250;
		for (int i = 0; i < numOfatoms; ++i) {
			Atom newAtom;
			
			double charge = 0;
			double mult = ((rand() % 10) + 1);
			if (i + 1 != numOfatoms) {
				mult *= ((rand() % 2) == 0) ? 1 : -1;
				charge = electronCharge * mult;
			}
			else {
				charge = -chargeSum;
				mult = charge / electronCharge;
			}
			chargeSum += charge;
			
			vec3 color = vec3(
				(mult < 0) ? 0.2 + (-mult / 10) : 0,
				0,
				(mult > 0) ? 0.2 + (mult / 10) : 0);

			bool badPos = true;
			vec2 pos;
			float r = hidrogenWeigth * (rand() % 25 + 10);

			while (badPos) {
				badPos = false;
				if (i == 0) {
					pos = vec2(firstX, firstY);
				}
				else {
					float rx = rand() % 301 - 150;
					float ry = rand() % 301 - 150;
					pos = vec2(firstX + rx, firstY + ry);
				}

				for (int j = 0; j < i; j++) {
					if (distance(atoms[j].getPos(), pos) < atoms[j].getR() + r) {
						badPos = true;
						j = i;
					}
				}
			}
			
			sumOfMass += r;
			newAtom.create(pos, color, r, charge);
			atoms[i] = newAtom;
			if (i > 0) {
				Line l;
				int closest = 0;
				float minDist = 1200;
				for (int j = 0; j < i; j++) {
					float dist = distance(atoms[j].getPos(), pos);
					if (dist < minDist) {
						closest = j;
						minDist = dist;
					}
				}
				l.create(atoms[closest].getPos(), atoms[i].getPos(), vec3(1, 1, 1));
				lines[i] = l;
			}
		}
		center = centerOfMass();

		for (int i = 0; i < numOfatoms; i++) {
			rVec[i] = atoms[i].getPos() - center;
			atoms[i].setMolCenter(center);
		}

		for (int i = 1; i < numOfatoms; i++) {
			lines[i].setMolCenter(center);
		}
	}

	void Draw() {
		for (int i = 0; i < numOfatoms; ++i) {
			atoms[i].Draw();
			if (i != 0) {
				lines[i].Draw();
			}
		}
	}

	void Animate(vec2 r, float omega) {
		for (int i = 0; i < numOfatoms; ++i) {
			atoms[i].Animate(r, omega);
			if (i != 0) {
				lines[i].Animate(r, omega);
			}
		}
		center = centerOfMass();
		theta = 0;
		for (int i = 0; i < numOfatoms; ++i) {
			rVec[i] = atoms[i].getPos() - center;
			theta += atoms[i].getR() * length(rVec[i]) * length(rVec[i])/10000;
		}
		
	}

	float drag() {
		return 0.00001 * length(vel);
	}

	vec2 centerOfMass() {
		vec2 res = vec2(0, 0);
		for (int i = 0; i < numOfatoms; i++) {
			Atom a = atoms[i];
			res = res + (a.getPos() * a.getR());
		}
		res = res / sumOfMass;
		return res;
	}

	vec2 columbForce2D(Atom a1, Atom a2) {
		vec2 ev = normalize(a1.getPos() - a2.getPos());
		return ev * a1.getCharge() * a2.getCharge() / (2 * M_PI * eps0 * distance(a1.getPos(), a2.getPos()));
	}

	void calcAnimate(Molekule m2) {
		float dt = 0.01;
		float omega1 = 0;
		vec2 v1 = vec2(0, 0);
		vec2 r1 = vec2(0, 0);
		float alpha1 = 0;
		for (int i = 0; i < numOfatoms; i++) {
			vec2 cf1_2 = vec2(0, 0);
			sumM = vec3(0, 0, 0);
			sumF = vec2(0, 0);
			float m_1 = atoms[i].getR();
			for (int j = 0; j < m2.numOfatoms; j++) {
				cf1_2 = 15000 * columbForce2D(atoms[i], m2.atoms[j]);
				sumM = sumM + cross(rVec[i], cf1_2);
				sumF = sumF + dot(rVec[i], cf1_2) * normalize(rVec[i]);
			}
			omega1 += ((sumM.z * 0.01 - drag()) / theta * dt);
			omega = omega1;
			alpha1 += (omega1 * dt);
			v1 = v1 + ((sumF - vel) / m_1 * dt);
			r1 = r1 + (v1 * dt);
			vel = v1;
		}
		Animate(r1, alpha1);
	}
};

Molekule m1, m2;
Line l;
Atom a;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	m1.create();
	m2.create();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(128.0/255, 128.0/255, 128.0/255, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m1.Draw();
	m2.Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
		case 27:  exit(0);				   break; //esc
		case 32:  onInitialization();	   break; //space
		case 'd': camera.Pan(vec2(-0.1, 0)); break;
		case 's': camera.Pan(vec2(0.1, 0));  break;
		case 'e': camera.Pan(vec2(0, -0.1)); break;
		case 'x': camera.Pan(vec2(0, 0.1));  break;
		case 'z': camera.Zoom(1.1);		   break;
		case 'u': camera.Zoom(0.9);		   break;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

float prevTime = 0.0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float timePassed = (time - prevTime)/1000;
	for (float i = 0; i < timePassed; i += 0.01) {
		m1.calcAnimate(m2);
		m2.calcAnimate(m1);
	}
	prevTime = time;
	glutPostRedisplay();
}