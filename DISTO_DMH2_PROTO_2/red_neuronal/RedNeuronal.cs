using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DISTO_DMH2_PROTO_2.red_neuronal
{
    public abstract class RedNeuronal
    {
        protected int longitud_Xt = 300;

        protected float[] Sigmoide(float[] suma)
        {
            float[] v = new float[suma.Length];
            for (int i = 0; i < suma.Length; i++)
            {
                v[i] = (float)(1 / (1 + Math.Exp(-suma[i])));
            }
            return v;
        }

        protected float[] productoHadamard(float[] h, float[] rt)
        {
            float[] v = new float[h.Length];
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = h[i] * rt[i];
            }
            return v;
        }

        protected float[] tangenteHiperbolica(float[] x)
        {
            float[] y = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = (float)Math.Tanh(x[i]);
            }
            return y;
        }

        protected List<float> SoftMax(List<float> entrada)
        {
            float suma = 0;
            foreach (float n in entrada)
            {
                suma += (float)Math.Exp(n);
            }
            List<float> prob = new List<float>();
            foreach (float n2 in entrada)
            {
                prob.Add((float)Math.Exp(n2) / suma);
            }
            return prob;
        }

        protected float[] softmax(float[] inputSoftmax)
        {
            float[] res = new float[inputSoftmax.Length];
            double suma = 0;
            for (int i = 0; i < inputSoftmax.Length; i++)
            {
                suma += Math.Exp(inputSoftmax[i]);
            }
            for (int j = 0; j < inputSoftmax.Length; j++)
            {
                res[j] = (float)(Math.Exp(inputSoftmax[j]) / suma);
            }
            return res;
        }

        protected float[] productoHadamard(float valor, float[] d3)
        {
            float[] nuevoVector = new float[d3.Length];
            for (int i = 0; i < nuevoVector.Length; i++)
            {
                nuevoVector[i] = valor * d3[i];
            }
            return nuevoVector;
        }

        protected float[,] matrizTraspuesta(float[,] matriz)
        {
            float[,] mt = new float[matriz.GetLength(0), matriz.GetLength(1)];
            for (int i = 0; i < matriz.GetLength(0); i++)
            {
                for (int j = 0; j < matriz.GetLength(1); j++)
                {
                    mt[j, i] = matriz[i, j];
                }
            }
            return mt;
        }

        protected float[,] multiplicarVectores(float[] vec1, float[] vec2)
        {
            float[,] matriz = new float[vec1.Length, vec2.Length];

            for (int i = 0; i < vec1.Length; i++)
            {
                for (int j = 0; j < vec2.Length; j++)
                {
                    matriz[i, j] = vec1[i] * vec2[j];
                }
            }
            return matriz;
        }

        protected float[] sumaVectores(float[] wx, float[] uh)
        {
            float[] vectorFinal = new float[longitud_Xt];
            for (int i = 0; i < wx.Length; i++)
            {
                vectorFinal[i] = wx[i] + uh[i];
            }
            return vectorFinal;
        }

        protected float[,] sumaMatrices(float[,] m1, float[,] m2)
        {
            float[,] res = new float[300, 300];
            for (int i = 0; i < m1.GetLength(0); i++)
            {
                for (int j = 0; j < m1.GetLength(1); j++)
                {
                    res[i, j] = m1[i, j] + m1[i, j];
                }
            }
            return res;
        }

        protected float[] Resta(float[] zt)
        {
            float[] resta = new float[longitud_Xt];
            for (int i = 0; i < zt.Length; i++)
            {
                resta[i] = 1 - zt[i];
            }
            return resta;
        }

        protected float[] multiplcarMatrizVector(float[,] w, float[] x)
        {
            float[] vectorFinal = new float[longitud_Xt];
            for (int fila = 0; fila < longitud_Xt; fila++)
            {
                //vectorFinal[fila] = 0;
                for (int col = 0; col < longitud_Xt; col++)
                {
                    vectorFinal[fila] += w[fila, col] * x[col];
                }
            }
            return vectorFinal;
        }

        protected float[] restaVectores(float[] v1, float[] v2)
        {
            float[] vf = new float[v1.Length];
            for (int i = 0; i < v1.Length; i++)
            {
                vf[i] = v1[i] - v2[i];
            }
            return vf;
        }
    }
}
