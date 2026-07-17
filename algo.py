def get_wheel_speeds(pid_output, error, found, medium_base_speed=0.6, speed_reduction_factor=0.5):
    """
    Рассчитывает нормализованные скорости колес [-1.0, 1.0].
    Включает режим поиска (вращение на месте), если линия потеряна (error == 0).
    """
    # 1. Режим поиска: если ошибка ровно 0, считаем, что линия потеряна
    if not found:
        # Крутимся на месте вправо (левое колесо вперед, правое назад)
        # Скорость 0.5 (50%), чтобы вращение было плавным и робот успел "заметить" линию
        return 0, 0

        # 2. Адаптивное торможение: чем выше ошибка, тем сильнее падает базовая скорость
    base_speed = medium_base_speed - abs(pid_output) * speed_reduction_factor

    # Гарантируем, что базовая скорость не станет отрицательной
    base_speed = max(0.0, base_speed)

    # 3. Дифференциальное руление
    left_speed = base_speed - pid_output
    right_speed = base_speed + pid_output

    # 4. Нормализуем результат для среды (от -1.0 до 1.0)
    left_speed = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))

    return left_speed, right_speed