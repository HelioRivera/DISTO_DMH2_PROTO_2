using DISTO_DMH2_PROTO_2.Registro;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;


namespace DISTO_DMH2_PROTO_2.red_neuronal
{
    class EntrenamientoGru : RedNeuronal
    {
        private string ruta_dataset_entrenamiento = @"./src/Data/dataset_entrenamiento.txt";
        private RegistroVectores registroVectores = new RegistroVectores();
        private Gru GRU;
        private Gru GRUaux;
        private List<Gru> redDesplegada;
        private List<float[]> listaEntradas;
        private List<float[]> listaSalidasEsperadas;
        private List<float[]> listaSalidasObtenidas;
        private List<string> listaDatasetEntrenamiento;
        private const float factorApendizaje = 0.03f;
        private const float errorMinimo = 0.000000001f;
        private const int cantidadEpocas = 5;


        public EntrenamientoGru(Gru G)
        {
            // entra una red entrenada o inicializada
            GRU = G;
            // GRUaux = new RedNeuronalRecurrente(false);
            listaDatasetEntrenamiento = cargarDatosEntrenamiento();
            listaEntradas = new List<float[]>();
            listaSalidasEsperadas = new List<float[]>();
            listaSalidasObtenidas = new List<float[]>();
            redDesplegada = new List<Gru>();
        }

        public void Entrenar()
        {
            float errorPromedioDataSet = 99;
            for (int epoca = 0; epoca < cantidadEpocas && errorPromedioDataSet > errorMinimo; epoca++)
            {//iterar por epocas
                errorPromedioDataSet = 0;
                Console.WriteLine("-------------------Epoca:" + epoca + "----------------------");
                foreach (string oracion in listaDatasetEntrenamiento)
                {
                    string[] oracionSplit = oracion.Split();
                    for (int i = 0; i < oracionSplit.Length - 1; i++)
                    {
                        listaEntradas.Add(registroVectores.obtenerVector(oracionSplit[i]));
                        listaSalidasEsperadas.Add(registroVectores.obtenerVector(oracionSplit[i + 1]));
                    }
                    float errorOracion = 0;
                    for (int i = 0; i < this.listaEntradas.Count; i++)
                    {
                        float[] vectorProbabilidades = GRU.feedForward(this.listaEntradas[i]);
                        //test
                        float max = vectorProbabilidades.Max();
                        int maxIndex = Array.IndexOf(vectorProbabilidades, max);
                        float[] vectorObtenido = registroVectores.obtenerVector(maxIndex);
                        //test
                        listaSalidasObtenidas.Add(vectorObtenido);
                        errorOracion += calcularError(vectorObtenido, listaSalidasEsperadas[i]);//va acumulando el error de cada palabra de la oracion
                        redDesplegada.Add(GRU); //va guardando cada estado de la GRU en cada tiempo
                    }
                    errorOracion = errorOracion / this.listaEntradas.Count; // promedio de error de la oracion
                    Console.WriteLine("Error Oracion:" + errorOracion);
                    errorPromedioDataSet += errorOracion; // va a cumulando el error de cada oracion
                    TBPTT();
                    this.redDesplegada.Clear(); // limpia los timestep
                    this.listaEntradas.Clear(); // limpia la lista de entradas
                    this.listaSalidasEsperadas.Clear(); // limpia la lista de salidas esperadas
                    this.listaSalidasObtenidas.Clear();
                }
                errorPromedioDataSet = errorPromedioDataSet / this.listaDatasetEntrenamiento.Count; // promedio de error del dataset
                Console.WriteLine("Error Dataset:" + errorPromedioDataSet);
            }
        }

        private void TBPTT()
        {
            float[,] dLrespectodV = new float[GRU.getX().Length, GRU.getX().Length];
            float[,] dLrespectodWh = new float[GRU.getX().Length, GRU.getX().Length];
            float[,] dLrespectodUh = new float[GRU.getX().Length, GRU.getX().Length];
            float[] dLrespectodR = new float[GRU.getX().Length];
            float[,] dLrespectodWr = new float[GRU.getX().Length, GRU.getX().Length];
            float[,] dLrespectodUr = new float[GRU.getX().Length, GRU.getX().Length];
            float[] dLrespectodZ = new float[GRU.getX().Length];
            float[,] dLrespectodWz = new float[GRU.getX().Length, GRU.getX().Length];
            float[,] dLrespectodUz = new float[GRU.getX().Length, GRU.getX().Length];
            float[] dLrespectodBiasZ = new float[GRU.getX().Length];
            float[] dLrespectodBiasR = new float[GRU.getX().Length];
            float[] dLrespectodBiasH = new float[GRU.getX().Length];
            float[] dLrespectodBiasV = new float[GRU.getX().Length];
            for (int i = this.redDesplegada.Count - 1; i > 1; i--)
            {
                Console.WriteLine(i);
                // float[] qt = calculoQ(
                //     redDesplegada[i].Uh,
                //     redDesplegada[i].X,
                //     redDesplegada[i].Wh,
                //     redDesplegada[i - 1].S,
                //     redDesplegada[i].R); // nose si se vaya a usar
                float[] sigma = calculoSigma(this.listaSalidasObtenidas[i],
                                             this.listaSalidasEsperadas[i],
                                             redDesplegada[i].getV(),
                                             redDesplegada[i].getS()); // SUUUMMMMAAAATTOOOORRRIIIAAAAA???????
                dLrespectodV = calculodLrespectodV(dLrespectodV,
                                                   this.listaSalidasObtenidas[i],
                                                   this.listaSalidasEsperadas[i],
                                                   redDesplegada[i].getS()); // se va sumando
                dLrespectodWh = calculodLrespectodWh(dLrespectodWh,
                                                     sigma,
                                                     this.redDesplegada[i - 1].getS(),
                                                     this.redDesplegada[i].getR());
                dLrespectodUh = calculodLrespectodUh(dLrespectodUh,
                                                     sigma,
                                                     redDesplegada[i - 1].getX());
                dLrespectodR = calculodLrespectodR(dLrespectodR,
                                                   sigma,
                                                   redDesplegada[i].getWr(),
                                                   redDesplegada[i - 1].getS());
                dLrespectodWr = calculodLrespectodWr(dLrespectodWr,
                                                     dLrespectodR,
                                                     redDesplegada[i].getR(),
                                                     redDesplegada[i - 1].getS());
                dLrespectodUr = calculodLrespectodUr(dLrespectodUr,
                                                     dLrespectodR,
                                                     redDesplegada[i].getR(),
                                                     redDesplegada[i].getX());
                dLrespectodZ = calculodLrespectodZ(dLrespectodZ,
                                                   sigma,
                                                   redDesplegada[i].getWz(),
                                                   redDesplegada[i].getR(),
                                                   redDesplegada[i - 2].getS(),
                                                   redDesplegada[i - 1].getH());
                dLrespectodWz = calculodLrespectodWz(dLrespectodWz,
                                                     dLrespectodZ,
                                                     redDesplegada[i].getZ(),
                                                     redDesplegada[i - 1].getS());
                dLrespectodUz = calculodLrespectodUz(dLrespectodUz,
                                                     dLrespectodZ,
                                                     redDesplegada[i].getZ(),
                                                     redDesplegada[i].getX());

                dLrespectodBiasZ = calculodLrespectoBiasZyR(dLrespectodBiasZ,
                                                            redDesplegada[i].getUz(),
                                                            redDesplegada[i].getX(),
                                                            redDesplegada[i].getWz(),
                                                            redDesplegada[i - 1].getS(),
                                                            redDesplegada[i].getbiasZ());
                dLrespectodBiasR = calculodLrespectoBiasZyR(dLrespectodBiasR,
                                                          redDesplegada[i].getUr(),
                                                          redDesplegada[i].getX(),
                                                          redDesplegada[i].getWr(),
                                                          redDesplegada[i - 1].getS(),
                                                          redDesplegada[i].getbiasR());
                dLrespectodBiasH = calculodLrespectoBiasH(dLrespectodBiasH,
                                                          redDesplegada[i].getUh(),
                                                          redDesplegada[i].getX(),
                                                          redDesplegada[i].getWh(),
                                                          redDesplegada[i - 1].getS(),
                                                          redDesplegada[i].getZ(),
                                                          redDesplegada[i].getbiasH());
                dLrespectodBiasV = calculodLrespectoBiasV(dLrespectodBiasV,
                                                          redDesplegada[i].getV(),
                                                          redDesplegada[i].getS(),
                                                          redDesplegada[i].getbiasV());
            }
            GRU = GRUaux; // limpia las puertas
            this.GRU.setWh(calcularNuevoPesos(dLrespectodWh, this.GRU.getWh()));
            this.GRU.setUh(calcularNuevoPesos(dLrespectodUh, this.GRU.getUh()));
            this.GRU.setWr(calcularNuevoPesos(dLrespectodWr, this.GRU.getWr()));
            this.GRU.setUr(calcularNuevoPesos(dLrespectodUr, this.GRU.getUr()));
            this.GRU.setWz(calcularNuevoPesos(dLrespectodWz, this.GRU.getWz()));
            this.GRU.setUz(calcularNuevoPesos(dLrespectodUz, this.GRU.getUz()));
            this.GRU.setV(calcularNuevoPesos(dLrespectodV, this.GRU.getV()));
            this.GRU.setbiasZ(calcularNuevoPesosBias(dLrespectodBiasZ, this.GRU.getbiasZ()));
            this.GRU.setbiasR(calcularNuevoPesosBias(dLrespectodBiasR, this.GRU.getbiasR()));
            this.GRU.setbiasH(calcularNuevoPesosBias(dLrespectodBiasH, this.GRU.getbiasH()));
            this.GRU.setbiasV(calcularNuevoPesosBias(dLrespectodBiasV, this.GRU.getbiasV()));
            GRUaux = GRU;
        }

        private float[] calcularNuevoPesosBias(float[] biasGrad, float[] vectorBias)
        {
            float[] mult = new float[vectorBias.Length];
            for (int i = 0; i < biasGrad.Length; i++)
                mult[i] = -1 * factorApendizaje * biasGrad[i];
            return sumaVectores(biasGrad, mult);
        }

        private float[] calculodLrespectoBiasV(float[] dLrespectodBiasV, float[,] v, float[] s, float[] biasV)
        {
            float[] m = multiplcarMatrizVector(v, s);
            float[] act = softmax(m);
            float[] res = new float[m.Length];
            for (int i = 0; i < m.Length; i++)
            {
                res[i] = act[i] * (1 - act[i]);
            }
            return sumaVectores(res, dLrespectodBiasV);
        }

        private float[] calculodLrespectoBiasH(float[] dLrespectodBiasH, float[,] uh, float[] x, float[,] wh, float[] s, float[] z, float[] bias)
        {
            float[] mUX = multiplcarMatrizVector(uh, x);
            float[] mWS = multiplcarMatrizVector(wh, s);
            float[] ph = productoHadamard(mWS, z);
            float[] sum = sumaVectores(mUX, ph);
            float[] sumaBias = sumaVectores(sum, bias);
            float[] dTanH = new float[300];
            for (int i = 0; i < sumaBias.Length; i++)
            {
                dTanH[i] = (float)(1 - Math.Pow(Math.Tanh(sumaBias[i]), 2));
            }
            return sumaVectores(dTanH, dLrespectodBiasH);
        }

        private float[] calculodLrespectoBiasZyR(float[] dLrespectodBias, float[,] U, float[] X, float[,] W, float[] S, float[] bias)
        {
            float[] mUX = multiplcarMatrizVector(U, X);
            float[] mWS = multiplcarMatrizVector(W, S);
            float[] sumaVec = sumaVectores(mUX, mWS);
            float[] sumaBias = sumaVectores(sumaVec, bias);
            float[] res1 = new float[300];
            float[] res2 = new float[300];
            float[] dSigmoide = new float[300];
            for (int i = 0; i < bias.Length; i++)
            {
                res1[i] = (float)Math.Exp(-1 * sumaBias[i]);
                res2[i] = (float)Math.Pow(1 + Math.Exp(-1 * sumaBias[i]), 2);
                dSigmoide[i] = (res1[i] / res2[i]);
            }
            return sumaVectores(dLrespectodBias, dSigmoide);
        }
        private float[,] calcularNuevoPesos(float[,] matrizGrad, float[,] matriz)
        {
            float[,] mult = new float[matriz.GetLength(0), matriz.GetLength(1)];
            for (int i = 0; i < matriz.GetLength(0); i++)
                for (int j = 0; j < matriz.GetLength(1); j++)
                    mult[i, j] = -1 * factorApendizaje * matrizGrad[i, j];
            return sumaMatrices(matriz, mult);
        }

        private float[] calculoQ(float[,] uh, float[] x, float[,] wh, float[] s, float[] r)
        {
            float[] UhporX = multiplcarMatrizVector(uh, x);
            float[] ShadamR = productoHadamard(s, r);
            float[] WhporShadamR = multiplcarMatrizVector(wh, ShadamR);
            return sumaVectores(UhporX, WhporShadamR);
        }

        private float[] calculoSigma(float[] palabraObtenida, float[] palabraEsperada, float[,] v, float[] s)
        {
            // (ygorro - yt)*V*(1-Zt) probar con Zt y Rt
            float[] restaPalabras = restaVectores(palabraObtenida, palabraEsperada);
            for (int i = 0; i < s.Length; i++)
                s[i] = s[i] * s[i];//si
            float[] restaS = Resta(s);
            float[] multResV = multiplcarMatrizVector(v, restaPalabras);
            return productoHadamard(restaS, multResV);
        }

        private float[,] calculodLrespectodV(float[,] dLrespectodV, float[] palabraObtenida, float[] palabraEsperada, float[] s)
        {
            float[] resta = restaVectores(palabraObtenida, palabraEsperada);
            float[,] pe = multiplicarVectores(resta, s);
            return sumaMatrices(dLrespectodV, pe);
        }

        private float[,] calculodLrespectodWh(float[,] dLrespectodWh, float[] sigma, float[] s, float[] r)
        {
            float[] ph = productoHadamard(s, r);
            float[,] dWh = multiplicarVectores(ph, sigma);
            return sumaMatrices(dLrespectodWh, dWh);
        }

        private float[,] calculodLrespectodUh(float[,] dLrespectodUh, float[] sigma, float[] x)
        {
            return sumaMatrices(multiplicarVectores(x, sigma),
                                       dLrespectodUh);
        }

        private float[] calculodLrespectodR(float[] dLrespectodR, float[] sigma, float[,] wr, float[] s)
        {
            //Wr debe ser traspuesta
            float[,] Wrtrasp = matrizTraspuesta(wr);
            float[] sighadamS = productoHadamard(sigma, s);
            float[] mult = multiplcarMatrizVector(Wrtrasp, sighadamS);
            return sumaVectores(dLrespectodR, mult);
        }

        private float[,] calculodLrespectodWr(float[,] dLrespectodWr, float[] dLrespectodR, float[] r, float[] s)
        {
            float[] ph1 = productoHadamard(dLrespectodR, r);
            float[] ph2 = productoHadamard(ph1, Resta(r));
            float[,] pe = multiplicarVectores(ph2, s);
            return sumaMatrices(pe, dLrespectodWr);
        }

        private float[,] calculodLrespectodUr(float[,] dLrespectodUr, float[] dLrespectodR, float[] r, float[] x)
        {
            float[] ph1 = productoHadamard(dLrespectodR, r);
            float[] ph2 = productoHadamard(ph1, Resta(r));
            float[,] pe = multiplicarVectores(ph2, x);
            return sumaMatrices(pe, dLrespectodUr);
        }

        private float[] calculodLrespectodZ(float[] dLrespectodZ, float[] sigma, float[,] wz, float[] r, float[] s, float[] h)
        {
            //wz debe ser traspuesta
            float[,] trasp = matrizTraspuesta(wz);
            float[] resta = restaVectores(s, h);
            float[] hadam1 = productoHadamard(resta, r);
            float[] hadam2 = productoHadamard(sigma, hadam1);
            float[] mult = multiplcarMatrizVector(trasp, hadam2);
            return sumaVectores(dLrespectodZ, mult);
        }

        private float[,] calculodLrespectodWz(float[,] dLrespectodWz, float[] dLrespectodZ, float[] z, float[] s)
        {
            float[] hadam1 = productoHadamard(dLrespectodZ, z);
            float[] resta = Resta(z);
            float[] hadam2 = productoHadamard(resta, hadam1);
            float[,] mult = multiplicarVectores(hadam2, s);
            return sumaMatrices(dLrespectodWz, mult);
        }

        private float[,] calculodLrespectodUz(float[,] dLrespectodUz, float[] dLrespectodZ, float[] z, float[] x)
        {
            float[] ph1 = productoHadamard(dLrespectodZ, z);
            float[] ph2 = productoHadamard(ph1, Resta(z));
            float[,] pe = multiplicarVectores(ph2, x);
            return sumaMatrices(pe, dLrespectodUz);
        }

        private float calcularError(float[] vectorObtenido, float[] vectorEsperado)
        {
            float error = 0;
            for (int i = 0; i < vectorEsperado.Length; i++)
            {
                error += (float)Math.Pow(vectorObtenido[i] - vectorEsperado[i], 2);
            }
            float errorTotal = error / vectorEsperado.Length;
            return errorTotal;
        }

        private List<string> cargarDatosEntrenamiento()
        {
            listaDatasetEntrenamiento = new List<string>();
            if (File.Exists(ruta_dataset_entrenamiento))
            {
                IEnumerable<string> lines = File.ReadLines(ruta_dataset_entrenamiento);
                for (int i = 0; i < lines.Count(); i++)
                {
                    listaDatasetEntrenamiento.Add(lines.ElementAt(i));
                }
            }
            return listaDatasetEntrenamiento;
        }
    }
}
