using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DISTO_DMH2_PROTO_2.Registro
{
    class RegistroPalabrasProblematicas
    {
        private string ruta_dataset_palabrasproblematicas = @"./src/Data/dataset_palabrasproblematicas.txt";
        private List<string> palabrasProblematicas;

        public RegistroPalabrasProblematicas()
        {
            cargarRegistro();
        }
        private void cargarRegistro()
        {
            palabrasProblematicas = new List<string>();
            if (File.Exists(ruta_dataset_palabrasproblematicas))
            {
                var fileContent = File.ReadAllText(ruta_dataset_palabrasproblematicas);
                var array = fileContent.Split((string[])null, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < array.Length; i++)
                {
                    if (!palabrasProblematicas.Contains(array[i].ToLower()))
                    {
                        palabrasProblematicas.Add(array[i].ToLower());
                    }
                }
                Console.WriteLine("Registro Palabras problematicas cargado exitosamente");
            }
            else Console.WriteLine("NO EXISTE EL datasetPalabrasProblematicas");
        }
        public bool esPalabraProblematica(string palabra)
        {
            if (palabrasProblematicas.Contains(palabra))
                return true;
            else return false;
        }
        public void agregarPalabraProblematica(string palabra)
        {
            if (!palabrasProblematicas.Contains(palabra))
                this.palabrasProblematicas.Add(palabra);
        }
        public void eliminarPalabraProblematica(string palabra)
        {
            if (this.palabrasProblematicas.Contains(palabra))
            {
                this.palabrasProblematicas.Remove(palabra);
            }
            else Console.WriteLine("La palabra no se encuentra en el registro");
        }
        public void guardarRegistro()
        {
            System.IO.FileStream f = System.IO.File.Create(ruta_dataset_palabrasproblematicas);
            f.Close();
            using (System.IO.StreamWriter sw = System.IO.File.AppendText(ruta_dataset_palabrasproblematicas))
            {
                for (int i = 0; i < this.palabrasProblematicas.Count; i++)
                    sw.WriteLine(this.palabrasProblematicas[i]);
                Console.WriteLine("Registro Palabras problematicas guardado exitosamente");
                sw.Close();
            }
        }
        public List<string> getpalabrasProblematicas() { return this.palabrasProblematicas; }
    }
}
