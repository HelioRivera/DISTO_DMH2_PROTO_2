using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DISTO_DMH2_PROTO_2.red_neuronal
{
    class Gru : RedNeuronal
    {
        private const string ruta_pesosred = @"./src/Data/pesosred.txt";
 
        //Variables a utilizar en la arquitectura de red neuronal
        private float[,] Wz;
        private float[,] Wr;
        private float[,] Wh;
        private float[,] Uz;
        private float[,] Ur;
        private float[,] Uh;
        private float[,] V;
        private float[] biasZ;
        private float[] biasR;
        private float[] biasH;
        private float[] biasV;
        private float[] H; // h(t) = tanh(Uh X(t) + Wh s(t-1))
        private float[] X; // entrada
        private float[] Z; // z(t) puerta
        private float[] R; // r(t) puerta
        private float[] S; // s(t) 
        private float[] O; // softmax

        public Gru(Boolean cargarRed)
        {
            if (cargarRed)
            {
                cargarGRU();
            }
            else
            {
                inicializarGRU();
            }
        }

        public void inicializarGRU()
        {
            //matrices
            this.Wz = generarPeso();
            this.Wr = generarPeso();
            this.Wh = generarPeso();
            this.Uz = generarPeso();
            this.Ur = generarPeso();
            this.Uh = generarPeso();
            this.V = generarPeso();
            //bias 
            this.biasZ = generarVector("bias");
            this.biasR = generarVector("bias");
            this.biasH = generarVector("bias");
            this.biasV = generarVector("bias");
            //
            this.H = new float[biasZ.Length];//candidato a estado oculto
            this.X = new float[biasZ.Length];//vector de entrada a gru en el instante t
            this.Z = new float[biasZ.Length];//update gate
            this.R = new float[biasZ.Length];//reset gate
            this.S = generarVector("inicial"); // estado oculto
            this.O = new float[biasZ.Length];//salida de la celda 
        }

        private void cargarGRU()
        {
            //this.h = control.generarVector(random, "inicial");
            //metodo cargar ~ desde control
        }

        public float[] feedForward(float[] Xt)
        {
            this.X = Xt;//vector de entrada de la red
            this.Z = updateResetGate(this.Uz, this.X, this.Wz, this.S, this.biasZ);
            this.R = updateResetGate(this.Ur, this.X, this.Wr, this.S, this.biasR);
            this.H = contentGate(this.Uh, Xt, this.Wh, this.S, this.Z, this.biasH);
            this.S = HiddenGate(this.Z, this.H, this.S);
            float[] inputSoftmax = multiplcarMatrizVector(this.V, this.S);
            this.O = softmax(sumaVectores(inputSoftmax, this.biasV));
            //float max = this.O.Max();
            //int maxIndex = Array.IndexOf(this.O,max);
            return this.O;
        }

        private float[] updateResetGate(float[,] U, float[] X, float[,] W, float[] S, float[] bias)
        {
            float[] UporX = multiplcarMatrizVector(U, X);
            float[] WporH = multiplcarMatrizVector(W, S);
            float[] sumaUXWH = sumaVectores(UporX, WporH);
            float[] sumaBias = sumaVectores(sumaUXWH, bias);
            float[] sigmoide = Sigmoide(sumaBias);
            return sigmoide;
        }

        private float[] contentGate(float[,] Uh, float[] X, float[,] Wh, float[] S, float[] Zt, float[] bias)
        {
            float[] UporX = multiplcarMatrizVector(Uh, X);
            float[] prodHadSZt = productoHadamard(S, Zt);
            float[] WhporSZt = multiplcarMatrizVector(Wh, prodHadSZt);
            float[] suma = sumaVectores(UporX, WhporSZt);
            float[] suma2 = sumaVectores(suma, bias);
            float[] tanh = tangenteHiperbolica(suma2);
            return tanh;
        }

        private float[] HiddenGate(float[] Zt, float[] H, float[] S)
        {
            float[] resta = Resta(Zt);
            float[] ph1 = productoHadamard(resta, H);
            float[] ph2 = productoHadamard(Zt, S);
            float[] suma = sumaVectores(ph1, ph2);
            return suma;
        }

        private float[] generarVector(string tipo)
        {
            Random random = new Random();
            float[] vector = new float[longitud_Xt];
            for (int i = 0; i < vector.Length; i++)
            {
                switch (tipo)
                {
                    case "inicial": vector[i] = 0; break;
                    case "noInicial": vector[i] = ((float)random.NextDouble() - 0.4f) * 0.1f; break;
                    case "bias": vector[i] = ((float)random.NextDouble() - 0.6f); break;
                }
            }
            return vector;
        }

        public float[,] generarPeso()
        {
            Random random = new Random();
            float[,] peso = new float[longitud_Xt, longitud_Xt];
            for (int i = 0; i < longitud_Xt; i++)
            {
                for (int j = 0; j < longitud_Xt; j++)
                {
                    peso[i, j] = ((float)random.NextDouble() - 0.4f) * 0.1f;
                }
            }
            return peso;
        }

        // FUNCIONES NO UTILIZADAS
        public Dictionary<int, float[]> DiccionarioOneHot(int tamVocabulario)
        {
            Dictionary<int, float[]> diccionarioOneHot = new Dictionary<int, float[]>();
            for (int i = 0; i < tamVocabulario; i++)
            {
                float[] vecOneHot = new float[tamVocabulario];
                for (int j = 0; j < tamVocabulario; j++)
                {
                    if (j == i)
                    {
                        vecOneHot[j] = 1;
                    }
                    else
                    {
                        vecOneHot[j] = 0;
                    }
                }
                diccionarioOneHot.Add(i, vecOneHot);
            }
            return diccionarioOneHot;

        }

        public float[] SoftMaxToOneHot(float[] vectorSoftmax, int tamanovocabulario)
        {
            float[] onehot = new float[tamanovocabulario];
            for (int i = 0; i < tamanovocabulario; i++)
            {
                if (vectorSoftmax[i] > 0.5)
                {
                    onehot[i] = 1;
                }
                else
                {
                    onehot[i] = 0;
                }
            }
            return onehot;
        }

        public float productoPuntoVector(float[] vector)
        {
            float[] v1 = vector;
            float[] v2 = vector;
            float suma = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                suma = suma + (v1[i] * v2[i]);
            }
            return suma;
        }

        public float[,] inicializarMatriz()
        {
            float[,] unaMatriz = new float[longitud_Xt, longitud_Xt];
            for (int x = 0; x < unaMatriz.GetLength(0); x++)
            {
                for (int y = 0; y < unaMatriz.GetLength(1); y++)
                {
                    unaMatriz[x, y] = 0;
                }
            }
            return unaMatriz;
        }

        public float[] productoPuntoMatrizVector(float[,] matriz, float[] vector)
        {
            float[] vectorFinal = new float[vector.Length];
            for (int fila = 0; fila < matriz.GetLength(0); fila++)
            {
                vectorFinal[fila] = 0;
                for (int col = 0; col < matriz.GetLength(1); col++)
                {
                    vectorFinal[fila] += matriz[fila, col] * vector[col];
                }
            }
            return vectorFinal;
        }

        public float[,] resta1aMatriz(float[,] matriz)
        {
            float[,] m = new float[300, 300];
            for (int i = 0; i < matriz.GetLength(0); i++)
            {
                for (int j = 0; j < matriz.GetLength(1); j++)
                {
                    m[i, j] = 1 - matriz[i, j];
                }
            }
            return m;
        }

        public float[,] getWz() { return this.Wz; }
        public float[,] getWr() { return this.Wr; }
        public float[,] getWh() { return this.Wh; }
        public float[,] getUz() { return this.Uz; }
        public float[,] getUr() { return this.Ur; }
        public float[,] getUh() { return this.Uh; }
        public float[,] getV() { return this.V; }
        public float[] getbiasZ() { return this.biasZ; }
        public float[] getbiasR() { return this.biasR; }
        public float[] getbiasH() { return this.biasH; }
        public float[] getbiasV() { return this.biasV; }
        public float[] getH() { return this.H; } // h(t) = tanh(Uh X(t) + Wh s(t-1))
        public float[] getX() { return this.X; } // entrada
        public float[] getZ() { return this.Z; } // z(t) puerta
        public float[] getR() { return this.R; } // r(t) puerta
        public float[] getS() { return this.S; } // s(t) 
        public float[] getO() { return this.O; } // softmax
        public void setWz(float[,] nuevo) { this.Wz = nuevo; }
        public void setWr(float[,] nuevo) { this.Wr = nuevo; }
        public void setWh(float[,] nuevo) { this.Wh = nuevo; }
        public void setUz(float[,] nuevo) { this.Uz = nuevo; }
        public void setUr(float[,] nuevo) { this.Ur = nuevo; }
        public void setUh(float[,] nuevo) { this.Uh = nuevo; }
        public void setV(float[,] nuevo) { this.V = nuevo; }
        public void setbiasZ(float[] nuevo) { this.biasZ = nuevo; }
        public void setbiasR(float[] nuevo) { this.biasR = nuevo; }
        public void setbiasH(float[] nuevo) { this.biasH = nuevo; }
        public void setbiasV(float[] nuevo) { this.biasV = nuevo; }
    }
}
