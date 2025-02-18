#ifndef MYTYPES_H
#define MYTYPES_H

//#define HYDRODYN


//#define DIRECT_NBODY
//#define NMULTI 4
#define QUADRUPOLE
//#define NMULTI 10
#define OCTUPOLE
#define NMULTI 20
//#define HEXADECAPOLE
//#define NMULTI 35

double masspart;
double IRS;
double COEFF;
double softlen;
double param[4];

#define NSON 2
typedef unsigned long INT64;

typedef struct {
	int  updated;
	int  npart;
	int  son[NSON]; // all boxes are stecked level by level. child[box i] is in 2*i+(0,1)
	double split;
	double width[3];
	double center[3];
	double M[NMULTI];
	double L[NMULTI];
} Node;

typedef struct {
	int npart; //number of particles, all of which are stored sequentially in parts array
	int ipart; //index of particle in parts array
	double width[3];
	double center[3];
	double M[NMULTI];
	double L[NMULTI];

} Pack;



typedef struct {
	double pos[3];
	double acc[3];
	double vel[3];
    	double acc_pm[3];
//	double mass;
//	double len;
//    	long id;
//	int active;
} Body;

Body *part;
Node *btree;
Pack *leaf;


#endif /// MYTYPES_H ////


