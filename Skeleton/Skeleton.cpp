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
const int numOflinePoints = 100; 
//const double k = 8.988e9; //Coulomb number
const double hidrogenWeigth = 1.00797; // g/Mol
const double electronCharge = -1.602176634;//e-19;//e-19; //* 10 ^ (-19)coulomb
const double eps0 = 8.854187817;//e-12;

class Camera2D {
	vec2 wCenter;	//center in world coordinates
	vec2 wSize;		//width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(600, 600) {}

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;		//2D camera 
GPUProgram gpuProgram; // vertex and fragment shaders
//unsigned int vao;	   // virtual world on the GPU

float hiperbolicW(float x, float y) {
	return sqrt(x * x + y * y + 1);
}

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
		//wTranslate = t;		
		pos = t;
		color = c;
		sx = s;
		sy = s;
		r = s;
		charge = ch;

		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
		vec2 vertices[nv];
		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vertices[i] = vec2(cos(fi), sin(fi));
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			&vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		//STARTING POS
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");
	}

	vec2 getPos() {
		return pos;
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
		//sx = 10;
		//sy = 10;
		
		wTranslate = wTranslate + r;
		pos = pos + r;
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
			pos.x - mol_center.x, pos.y - mol_center.y, 0, 1);
		
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

		//COLOR
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

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
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
		vec2 vertices[numOflinePoints];
		vertices[0] = start;
		for (int i = 1; i < numOflinePoints - 1; i++) {
			vertices[i] = vec2(start + (i*(end-start)/numOflinePoints));			
		}
		vertices[numOflinePoints - 1] = end;

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * numOflinePoints,  // # bytes
			&vertices,	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		//STARTING POS
		mat4 MVPTransform = M() * camera.V() * camera.P();
		printf("KEZD************************************************************\n");
		gpuProgram.setUniform(MVPTransform, "MVP");
		printf("VEG************************************************************\n");
	}


	void Animate(vec2 r, float omega) {
		sx = 1;
		sy = 1;
		wTranslate = wTranslate + r;
		//start = start + r;
		//end = end + r;
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

		//COLOR
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

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
	float sumOfMass;
	vec2 rVec[8];
	vec3 sumM;
	vec2 sumF;


	Molekule() { 
		numOfatoms = 0; 
		sumOfMass = 0;
	}

	void create()  {		
		double chargeSum = 0;
		sumOfMass = 0;

		numOfatoms = rand() % 7 + 2; //between 2 and 8
		printf("\n\n**************\nNumber of atoms: %d\n", numOfatoms);
		float firstX = rand() % 601 - 300;
		float firstY = rand() % 601 - 300;
		for (int i = 0; i < numOfatoms; ++i) {
			Atom newAtom;
			
			double charge = 0;
			double mult = ((rand() % 10) + 1); //between 1 and 10
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
			float r = hidrogenWeigth * (rand() % 20 + 5);

			while (badPos) {
				badPos = false;
				if (i == 0) {
					pos = vec2(firstX, firstY);
				}
				else {
					float rx = rand() % 201 - 100;
					float ry = rand() % 201 - 100;
					pos = vec2(firstX + rx, firstY + ry);
				}

				//check if one atom overlaps with another
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
			printf("Atom %d: pos=(%f, %f)\n", i, pos.x, pos.y);
			printf("charge: %f    weight: %f\n", charge, r);
			printf("color: %.2f, %.2f, %.2f\n\n", color.x, color.y, color.z);
		}
		center = centerOfMass();

		for (int i = 0; i < numOfatoms; i++) {
			rVec[i] = atoms[i].getPos() - center;
			atoms[i].setMolCenter(center);
		}

		for (int i = 1; i < numOfatoms; i++) {
			lines[i].setMolCenter(center);
		}

		printf("Charge sum = %f\n", chargeSum);
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
		} //TODO
		center = centerOfMass();
		for (int i = 0; i < numOfatoms; ++i) {
			rVec[i] = atoms[i].getPos() - center;
		}
	}

	vec2 centerOfMass() {
		vec2 res = vec2(0, 0);
		for (int i = 0; i < numOfatoms; i++) {
			Atom a = atoms[i];
			res = res + (a.getPos() * a.getR());
		}
		res = res / sumOfMass;
		//printf("sum: %f %f\n", sum.x, sum.y);
		return res;
	}
};

Molekule m1, m2;
Line l;
Atom a;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	m1.create();
	m2.create();
	a.create(m1.center, vec3(0, 1, 0), 10, -1);
	//l.create(vec2(0,0), vec2(100,0), vec3(1,1,1));

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(128.0/255, 128.0/255, 128.0/255, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	m1.Draw();
	m2.Draw();
	a.Draw();
	//l.Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
	/*
	int numberOfMolecules = 2;
	for (int i = 0; i < numberOfMolecules; ++i) {
		double chargeSum = 0;

		float rx = (rand() % 30) / 15.0 - 1;
		float ry = (rand() % 30) / 15.0 - 1;
		//printf("%f\n", rx);
		int numberOfAtoms = rand() % 7 + 2; //between 2 and 8
		for (int j = 0; j < numberOfAtoms; ++j) {

			double mult = 0, charge = 0;
			if (j + 1 != numberOfAtoms) {
				mult = ((rand() % 10) + 1); //between 1 and 10
				if ((rand() % 2) == 0) {
					mult *= -1;
				}
				charge = electronCharge * mult;
			}
			else {
				charge = -chargeSum;
			}
			chargeSum += charge;
			
			// Set color to (0, 1, 0) = green
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location,(mult < 0) ? (-mult/10) : 0, 0, (mult > 0) ? (mult / 10) : 0); // 3 floats

			float MVPtransf[4][4] = { 0.1, 0, 0, 0,    // MVP matrix, 
									  0, 0.1, 0, 0,    // row-major!
									  0, 0, 1, 0,
									  rx+j*0.2, ry, 0, 1 };

			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

			glBindVertexArray(vao);  // Draw call
			glDrawArrays(GL_TRIANGLE_FAN, 0 startIdx, nv # Elements);
		}
		if (chargeSum != 0) {
			printf("ajajj");
		}
	}
	*/

}

// Key of ASCII code pressed
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

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

vec2 columbForce2D(Atom a1, Atom a2) {
	vec2 ev = normalize(a1.getPos() - a2.getPos());
	return ev * a1.getCharge() * a2.getCharge() / (2 * M_PI * eps0 * distance(a1.getPos(), a2.getPos()));
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float dt = 0.01;
	float theta1 = 10;
	float theta2 = 15;
	float omega1 = 0;
	float omega2 = 0;
	vec2 v1 = vec2(0, 0);
	vec2 v2 = vec2(0, 0);
	vec2 r1 = vec2(0, 0);
	vec2 r2 = vec2(0, 0);
	float alpha1 = 0;
	float alpha2 = 0;
	m1.sumF = vec2(0, 0);
	m1.sumM = vec3(0, 0, 0);
	m2.sumF = vec2(0, 0);
	m2.sumM = vec3(0, 0, 0);

	for (int i = 0; i < m1.numOfatoms; i++) { //egy molekula összes atomján végigmegyünk
		
		for (int j = 0; j < m2.numOfatoms; j++) {
			float m_1 = m1.atoms[i].getR();
			vec2 cf1_2 = columbForce2D(m1.atoms[i], m2.atoms[j]);
			m1.sumM = m1.sumM + cross(m1.rVec[i], cf1_2);
			m1.sumF = m1.sumF + dot(m1.rVec[i], cf1_2) * normalize(m1.rVec[i]);
			
			omega1 += (m1.sumM.z / theta1 * dt);
			alpha1 += (omega1 * dt);
			v1 = v1 + (m1.sumF / m_1 * dt);
			r1 = r1 + (v1 * dt);

			//**************************************
			/*
			float m_2 = m2.atoms[i].getR();
			vec2 cf2_1 = columbForce2D(m1.atoms[j], m2.atoms[i]);
			m2.sumM = m2.sumM + cross(m2.rVec[j], cf2_1);
			m2.sumF = m2.sumF + dot(m2.rVec[j], cf2_1) * normalize(m2.rVec[j]);

			omega2 += (m2.sumM.z / theta2 * dt);
			alpha2 += (omega2 * dt);
			v2 = v2 + (m2.sumF / m_2 * dt);
			r2 = r2 + (v2 * dt);*/

		}


	}
	printf("r: %f\n", r1.x);
	//m1.Animate(r1, omega1);
	//m2.Animate(r2, omega2);

	glutPostRedisplay();
}



vec2 drag(Molekule m) {
	return -5*m.vel;
}

/*theta = m * length(m1.getCenter()) * length(m1.getCenter());
			vec2 cf = columbForce2D(m1_atoms[i], m2_atoms[j]);
			vecM = cross(m1.getRVec()[j], cf);
			vec2 cent = m1.getCenter();
			float alpha = acos(dot(cent, cf)/(length(cent)*length(cf)));
				
			float alphaPart = alpha - floor(alpha);
			alpha = (int)alpha % 90 + alphaPart;
			sumF = sumF + sin((double)alpha) * cf;	
			sumM = sumM + cos(alpha) * cf;

			//vec2 cent = m1.centerOfMass();
			//vec2 rVec = m1_atoms[j].getPos() - cent;


/*
vec2 cf = columbForce2D(m1_atoms[i], m2_atoms[j]);
sumM = sumM + cross(m1.getRVec()[j], cf);
printf("r: %f\n", cross(m1.getRVec()[j], cf).z);

vec2 cf = columbForce2D(m1_atoms[i], m2_atoms[j]);
//printf("columb force : %f\n", cf.x);
sumM = sumM + cross(m1.getRVec()[i], cf);
//printf("sumM: %f", sumM.z);


float alpha = acos(dot(m1.getCenter(), cf) / (length(m1.getCenter()) * length(cf)));
float alphaPart = alpha - floor(alpha);
alpha = (int)floor(alpha) % 90 + alphaPart;
sumF = sumF + sin((double)alpha) * cf; */