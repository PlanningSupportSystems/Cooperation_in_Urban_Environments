#pragma once

#include <QTextEdit>

class OutputTextEdit : public QTextEdit
{
    Q_OBJECT

public:
    OutputTextEdit(QWidget* parent = nullptr);

signals:
    void returnPressed();

protected:
    void keyPressEvent(QKeyEvent* event) override;
};
