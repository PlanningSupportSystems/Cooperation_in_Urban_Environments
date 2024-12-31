#include "OutputTextEdit.h"
#include <QKeyEvent>

OutputTextEdit::OutputTextEdit(QWidget* parent)
    : QTextEdit(parent)
{
    setAcceptRichText(false); // Disable rich text editing
}

void OutputTextEdit::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
    {
        emit returnPressed(); // Emit returnPressed signal when Enter key is pressed
    }
    else
    {
        QTextEdit::keyPressEvent(event); // Let the base class handle other key events
    }
}
