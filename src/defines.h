//
// Created by jan on 5.7.2024.
//

#ifndef DEFINES_H
#define DEFINES_H

#ifdef __GNUC__

#define INTERNAL_MODULE_FUNCTION __attribute__((visibility("hidden")))

#endif

#ifndef INTERNAL_MODULE_FUNCTION
#define INTERNAL_MODULE_FUNCTION
#endif

#endif // DEFINES_H
