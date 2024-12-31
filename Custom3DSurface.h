#pragma once
#ifndef CUSTOM3DSURFACE_H
#define CUSTOM3DSURFACE_H

#include <QtDataVisualization/Q3DSurface>  // Include the required Qt header file
#include <QMouseEvent>                     // Include necessary Qt header file for mouse event
#include <QTextEdit>

// Include any other necessary header files for your class here

// Namespace declarations if needed
QT_BEGIN_NAMESPACE
// Add any necessary namespace declarations here
QT_END_NAMESPACE

class Custom3DSurface : public Q3DSurface
{
    Q_OBJECT
public:
    Custom3DSurface(QWidget* parent, QTextEdit* textEdit);

protected:
    void mousePressEvent(QMouseEvent* event) override;  // Override mouse press event

private:
    QTextEdit* m_textEdit;

    void handleMouseClick(const QPoint& pos);  // Private method for handling mouse click
    void setMessage(const QString& message);

};

#endif // CUSTOM3DSURFACE_H

