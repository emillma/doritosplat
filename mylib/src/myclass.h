#include <optix.h>

class OptixLink
{
private:
    int m_year{};
    int m_month{};
    int m_day{};

public:
    OptixLink();

    int getYear() const { return m_year; }
    int getMonth() const { return m_month; }
    int getDay() const { return m_day; }
};