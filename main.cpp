#include "URBAN_ABM_QT.h"
#include <QtWidgets/QApplication>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    URBAN_ABM_QT w;
    w.show();
    


    return a.exec();
}
