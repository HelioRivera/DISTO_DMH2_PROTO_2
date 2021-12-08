using DISTO_DMH2_PROTO_2.red_neuronal;
using System;

namespace DISTO_DMH2_PROTO_2
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Gru GRU = new Gru(false);
            EntrenamientoGru entrenar = new EntrenamientoGru(GRU);
            entrenar.Entrenar();
        }
    }
}
