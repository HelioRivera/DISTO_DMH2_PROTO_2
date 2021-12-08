using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DISTO_DMH2_PROTO_2.Registro
{
    class RegistroVectores
    {
        private string ruta_dataset_palabrasvectorizadas = @"./src/Data/dataset_palabrasvectorizadas.txt";
        private Dictionary<string, float[]> diccionarioStringVector;
        private Dictionary<float[], string> diccionarioVectorString;
        public Dictionary<int, float[]> vectoresIndexados;

        public RegistroVectores()
        {
            cargarRegistro();
        }
        private void cargarRegistro()
        {
            diccionarioStringVector = new Dictionary<string, float[]>();
            diccionarioVectorString = new Dictionary<float[], string>();
            vectoresIndexados = new Dictionary<int, float[]>();
            if (File.Exists(ruta_dataset_palabrasvectorizadas))
            {
                var fileContent = File.ReadAllText(ruta_dataset_palabrasvectorizadas);
                var array = fileContent.Split((string[])null, StringSplitOptions.RemoveEmptyEntries);
                float[] arregloAux = new float[300];
                float n;
                int j = 0;
                int numero = 0;
                for (int i = 0; i < array.Length; i++)
                {
                    if (float.TryParse(array[i], out n))
                    {
                        arregloAux[j] = n;
                    }
                    else
                    {
                        arregloAux = new float[300];
                        j = -1;
                        if (!diccionarioStringVector.ContainsKey(array[i].ToLower()))
                        {
                            diccionarioStringVector.Add(array[i].ToLower(), arregloAux);
                            diccionarioVectorString.Add(arregloAux, array[i].ToLower());
                            vectoresIndexados.Add(numero++, arregloAux);
                        }
                    }
                    j++;
                }
                Console.WriteLine("Registro Vectores cargado exitosamente" + diccionarioStringVector.Count);
            }
            else Console.WriteLine("NO EXISTE EL datasetPalabrasEspanol");
        }
        public float[] obtenerVector(string palabra)
        {
            if (diccionarioStringVector.ContainsKey(palabra.ToLower()))
            {
                return diccionarioStringVector[palabra.ToLower()];
            }
            else
            {
                Console.WriteLine("No se encontró la palabra " + palabra);
                return diccionarioStringVector["eso"];
            }
        }

        public float[] obtenerVector(int index)
        {
            return vectoresIndexados[index];
        }

        // metodo ahislado para obtener un vector exacto
        public string obtenerPalabra(float[] palabraPredichaVector)
        {
            List<float[]> listaVectores = new List<float[]>();
            List<float> listaErrores = new List<float>();

            for (int v = 0; v < this.diccionarioStringVector.Count; v++)
            {
                float[] vectorEsperado = this.vectoresIndexados[v];
                float error = 0;
                for (int i = 0; i < vectorEsperado.Length; i++)
                {
                    error += (float)Math.Pow(vectorEsperado[i] - palabraPredichaVector[i], 2);
                }
                float errorTotal = error / vectorEsperado.Length;
                listaVectores.Add(vectorEsperado);
                listaErrores.Add(errorTotal);
            }

            float errorMinimo = listaErrores.Min();
            int posicion = listaErrores.IndexOf(errorMinimo);


            return this.diccionarioVectorString[listaVectores[posicion]];
        }
    }
}
